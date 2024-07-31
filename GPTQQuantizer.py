from typing import Dict, List, Any, Optional, Callable, Type, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from quantizer import Quantizer
from GenericGPTQRunner import GenericGPTQRunner
from utils import *

def check_linear_int4_k(k, groupsize = 1, inner_k_tiles = None):
    k_divisible_by_groupsize = k % groupsize == 0
    if inner_k_tiles is not None:
        k_divisible_by_16_times_inner_k_tiles = k % (inner_k_tiles * 16) == 0
        return k_divisible_by_groupsize and k_divisible_by_16_times_inner_k_tiles
    return k_divisible_by_groupsize

def find_multiple(n: int, *args: Tuple[int]) -> int:
    k: int = reduce(lambda x, y: x * y // gcd(x, y), args + (1,))  # type: ignore[9]
    if n % k == 0:
        return n
    return n + k - (n % k)

def linear_forward_int4(
    x: torch.Tensor,
    weight_int4pack: torch.Tensor,
    scales_and_zeros: torch.Tensor,
    out_features: int,
    groupsize: int,
    precision: torch.dtype = torch.bfloat16,
    scales_precision: torch.dtype = torch.bfloat16,
):
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    c = torch.ops.aten._weight_int4pack_mm(
        x.to(precision),
        weight_int4pack,
        groupsize,
        scales_and_zeros.to(scales_precision)
    ).to(dtype=x.dtype)
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c

class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self, in_features: int, out_features: int,
        # TODO: remove dtype field, not used
        bias=False, device=None, dtype=None, groupsize: int = 128, inner_k_tiles: int = 8,
        precision: torch.dtype = torch.bfloat16, scales_precision: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.padding = not check_linear_int4_k(in_features, groupsize, inner_k_tiles)
        if self.padding:
            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)

        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.device = device
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.precision = precision
        self.scales_precision = scales_precision

        if dtype is not None:
            raise ValueError("Please specify 'precision' instead of 'dtype'")

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert in_features % (inner_k_tiles * 16) == 0, "require in_features % (innerKTiles * 16) == 0"
        self.register_buffer(
            "weight",
            torch.empty((out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2), dtype=torch.int32, device=device)
        )
        self.dtype = dtype
        self.register_buffer(
            "scales_and_zeros",
            torch.empty((in_features // groupsize, out_features, 2), dtype=self.scales_precision, device=device)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding:
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(
            input,
            self.weight,
            self.scales_and_zeros,
            self.out_features,
            self.groupsize,
            self.precision,
            self.scales_precision,
        )

def _replace_linear_int4(
    module: torch.nn.Module,
    groupsize: int,
    inner_k_tiles: Optional[int],
    padding_allowed: bool,
    skip_layer_func: Optional[Callable] = None,
    precision: torch.dtype = torch.bfloat16,
    scales_precision: torch.dtype = torch.bfloat16,
    linear_class: Type[torch.nn.Module] = WeightOnlyInt4Linear,
    copy_weights: bool = False,
):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and (skip_layer_func is None or not skip_layer_func(child.weight)):
            if check_linear_int4_k(child.in_features, groupsize, inner_k_tiles) or padding_allowed:
                new_linear = linear_class(
                    child.in_features,
                    child.out_features,
                    bias=False,
                    device=child.weight.device,
                    groupsize=groupsize,
                    inner_k_tiles=inner_k_tiles,
                    precision=precision,
                    scales_precision=scales_precision,
                )
                # TODO: merge with 8da4w?
                # In distributed training, the model may be instantiated
                # on the meta device, in which case there is no need to
                # copy the weights, and doing so will result in an error
                if copy_weights and child.weight.device != torch.device("meta"):
                    new_linear.weight = child.weight
                setattr(module, name, new_linear)
        else:
            _replace_linear_int4(
                child,
                groupsize,
                inner_k_tiles,
                padding_allowed,
                skip_layer_func,
                precision,
                scales_precision,
                linear_class,
                copy_weights,
            )


def replace_linear_int4(module, groupsize, inner_k_tiles, padding_allowed, skip_layer_func = None):
    _replace_linear_int4(
        module,
        groupsize,
        inner_k_tiles,
        padding_allowed,
        skip_layer_func,
        linear_class=WeightOnlyInt4Linear,
    )


class GPTQQuantizer(Quantizer):
    """
    This class implements a GPTQ Quantizer that can be used to apply GPTQ to a model in concert with the GenericGPTQRunner class.
    Unlike the base Quantizer class, the user does not need to implement the create_quantized_state_dict, instead they have to reimplement
    __init__ such that it defines the functions for the quantization mode. User is expected to reimplement convert_for_runtime.

    The following functions (which must be defined in __init__) are used to define the quantization mode for both GPTQ and
    create_quantized_state_dict. Here is a description of each function.

    get_qparams_func:
        A function that calculates the quantization qparams for an input tensor.
        Args:
            weight: A 2d weight tensor with non-integer dtype.
        Returns:
            qparams: it can have any format but will need to be handled by the other defined functions below.

    quantize_func:
        A function that applies quantization to an input tensor. It should be noted
        that this function needs to be able to handle quantizing the entire weight tensor, a single group,
        or a single column.
        Args:
            weight: A 2d weight tensor with non-integer dtype.
            qparams: the output from get_qparams_func
        Returns:
            quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)


    dequantize_func:
        A function that dequantizes an input quantized weight tensor. It should be noted
        that this function needs to be able to handle dequantizing the entire weight tensor, a single group,
        or a single column.
        Args:
            quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)
            qparams: the output from get_qparams_func
        Returns:
            weight: A 2d weight tensor with non-integer dtype.

    act_fake_quant_func (optional):
            A function that (dynamically) quantizes activation to input
            Args:
                input: input Tensor in f32/bf16/f16
            Returns:
                output: dynamically quantized and dequantized Tensor (with the same dtype as input)

    combine_qparams_list_func:
        A function that combines several qparams into one qparam.
        Args:
            qparams_list: a list of qparams objects, each obtained by calling get_qparams_func
            on a single group from a weight tensor
        Returns:
            qparams: an object of the same format as the qparams above.

    skip_layer_func:
        A function that determines which linear layers should be skipped during GPTQ
        Args:
            weight: A 2d weight tensor with non-integer dtype.
        Returns:
            skip: boolean indicating whether layer should be skipped

    make_names_and_values_dict_func:
        A function that prepares the qparams and quantized_weight and creates a dictionary indicating how they
        should be inserted into the state_dict. Generally any packing of the weight and qparams should be done here.
        Args:
            quantized_weight: A 2d quantized weight tensor (generally with an integer dtype)
            qparams: the output from get_qparams_func
        Returns:
            names_and_values_dict: a dictionary mapping the name of the parameters of the quantized module to the
            corresponding quantized weights and qparams.
    """

    def __init__(self):

        assert self.get_qparams_func is not None

        assert self.quantize_func is not None

        assert self.dequantize_func is not None

        assert self.combine_qparams_list_func is not None

        #  `make_names_and_values_dict_func`.
        assert self.make_names_and_values_dict_func is not None

    @torch.no_grad()
    def _create_quantized_state_dict(
        self,
        model,
        inputs,
        blocksize,
        percdamp,
        groupsize,
        #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
    ) -> Dict:
        print("Tracing model for GPTQ")
        GPTQ_runner = GenericGPTQRunner(
            model,
            inputs,
            blocksize,
            percdamp,
            groupsize,
        ).configure_quantization_mode(
            self.get_qparams_func,  # pyre-ignore[16]
            self.quantize_func,  # pyre-ignore[16]
            self.dequantize_func,  # pyre-ignore[16]
            self.combine_qparams_list_func,  # pyre-ignore[16]
            self.make_names_and_values_dict_func,  # pyre-ignore[16]
            self.skip_layer_func,  # pyre-ignore[16]
            self.act_fake_quant_func if hasattr(self, "act_fake_quant_func") else None,  # pyre-ignore[16]
        )
        print("Applying GPTQ to weights")
        GPTQ_runner.run()
        return GPTQ_runner.get_quantized_state_dict()

    def _convert_for_runtime(self, model: torch.nn.Module) -> "nn.Module":
        raise NotImplementedError("_convert_for_runtime not implemented")

    @torch.no_grad()
    def quantize(self, model: torch.nn.Module, inputs: List[torch.Tensor], **kwargs: Any) -> torch.nn.Module:
        pass

class Int4WeightOnlyGPTQQuantizer(GPTQQuantizer):
    def __init__(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=64,
        inner_k_tiles=8,
        padding_allowed=True,
        device: torch.device = torch.device("cuda"),
    ):
        self.blocksize = blocksize
        self.percdamp = percdamp
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding_allowed = padding_allowed
        self.device = device
        self.act_fake_quant_func = None
        n_bit = 4
        self.get_qparams_func = lambda w: get_groupwise_affine_qparams(
            w, n_bit, groupsize
        )
        self.quantize_func = lambda w, qparams: groupwise_affine_quantize_tensor_from_qparams(
            w, qparams[0], qparams[1], n_bit, groupsize
        )
        self.dequantize_func = lambda q, qparams: groupwise_affine_dequantize_tensor_from_qparams(
            q,
            qparams[0],
            qparams[1],
            n_bit,
            groupsize,
        )
        self.combine_qparams_list_func = lambda qparams_list: [
            torch.cat(x, dim=1) for x in zip(*qparams_list)
        ]
        # skip unless padding_allowed=True or its correctly sized
        self.skip_layer_func = lambda linear_weight: not (
            check_linear_int4_k(linear_weight.shape[-1], groupsize) or padding_allowed
        )

        # we need to do the padding here, both for q and the qparams if necessary

        # TODO: this is the gpt-fast version, merge with the main version later
        def make_names_and_values_dict_func(q, qparams):
            k = q.shape[1]
            if not check_linear_int4_k(k, groupsize):
                new_k = find_multiple(k, 1024)
            else:
                new_k = k
            # how much we need to pad the weight
            delta_k = new_k - q.shape[1]
            q = q.to(torch.int32).to(self.device)
            final_q = torch.ops.aten._convert_weight_to_int4pack(F.pad(q, pad=(0, delta_k)), inner_k_tiles)
            scales = qparams[0].to(torch.bfloat16).to(self.device)
            zeros = qparams[1].to(torch.bfloat16).to(self.device)
            scales_and_zeros = pack_tinygemm_scales_and_zeros(scales, zeros)
            # how many new groups we need for padded weight
            delta_groups = new_k // groupsize - scales_and_zeros.shape[0]
            final_s_and_z = F.pad(scales_and_zeros, pad=(0,0,0,0,0, delta_groups), value=1)
            return {"weight": final_q, "scales_and_zeros": final_s_and_z}

        self.make_names_and_values_dict_func = make_names_and_values_dict_func
        super().__init__()

    def _convert_for_runtime(self, model):
        replace_linear_int4(
            model,
            self.groupsize,
            self.inner_k_tiles,
            self.padding_allowed,
            skip_layer_func=self.skip_layer_func,
        )
        return model

    def quantize(self, model: torch.nn.Module, inputs: List[torch.Tensor], **kwargs: Any) -> torch.nn.Module:
        state_dict = self._create_quantized_state_dict(
            model,
            inputs,
            self.blocksize,
            self.percdamp,
            self.groupsize,
        )
        model = self._convert_for_runtime(model)
        model.load_state_dict(state_dict, strict=False)
        return model
