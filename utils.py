# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode
import torch.nn.utils.parametrize as parametrize
from quant_primitives import (
    MappingType,
    ZeroPointDomain,
    choose_qparams_affine,
    quantize_affine,
    dequantize_affine,
)

__all__ = [
    "compute_error",
    "_apply_logging_hook",
    "quantize_activation_per_token_absmax",
    "dynamically_quantize_per_channel",
    "dequantize_per_tensor",
    "dequantize_per_channel",
    "get_groupwise_affine_qparams",
    "pack_tinygemm_scales_and_zeros",
    "unpack_tinygemm_scales_and_zeros",
    "groupwise_affine_quantize_tensor_from_qparams",
    "groupwise_affine_dequantize_tensor_from_qparams",
    "groupwise_affine_quantize_tensor",
    "groupwise_affine_dequantize_tensor",
    "recommended_inductor_config_setter"
]

try:
    import lm_eval  # pyre-ignore[21]  # noqa: F401

    _lm_eval_available = True
except:
    _lm_eval_available = False

# basic SQNR
def compute_error(x, y):
    Ps = torch.linalg.norm(x)
    Pn = torch.linalg.norm(x - y)
    return 20 * torch.log10(Ps / Pn)


# logger for fqn + op + shape
# note: not safe for any kind of multithreading
_cur_fqn: Optional[str] = None


def _get_logging_hook(fqn):

    def forward_hook(module, input):
        global _cur_fqn
        _cur_fqn = fqn

    return forward_hook


def _apply_logging_hook(model):
    for name, mod in model.named_modules():
        mod.register_forward_pre_hook(_get_logging_hook(name))


# collections.defaultdict printing is weird with lambdas, so hand writing for now
_fqn_to_op_to_shape_to_count: Dict[
    Optional[str], Dict[Optional[str], Dict[Optional[str], int]]
] = {}


class LoggingTensorMode(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        rs = func(*args, **kwargs)
        global _cur_fqn
        op_name: str = f"{func.__module__}.{func.__name__}"
        shape_str = ""
        for arg in args:
            if isinstance(arg, torch.Tensor):
                shape_str += str(list(arg.shape)) + ", "
        if shape_str != "":
            shape_str = shape_str[:-2]

        if _cur_fqn not in _fqn_to_op_to_shape_to_count:
            _fqn_to_op_to_shape_to_count[_cur_fqn] = {}
        if op_name not in _fqn_to_op_to_shape_to_count[_cur_fqn]:
            _fqn_to_op_to_shape_to_count[_cur_fqn][op_name] = {}
        if shape_str not in _fqn_to_op_to_shape_to_count[_cur_fqn][op_name]:
            _fqn_to_op_to_shape_to_count[_cur_fqn][op_name][shape_str] = 0
        _fqn_to_op_to_shape_to_count[_cur_fqn][op_name][shape_str] += 1

        return rs

class _MultiInput:

    def __init__(self, inputs):

        self.values = list(inputs)

    def add_input(self, input):
        self.values.append(input)
        return self

    def __getitem__(self, slice):
        return _MultiInput(self.values[slice])

    def cuda(self):
        self.values = [
            val.cuda() if isinstance(val, torch.Tensor) else val for val in self.values
        ]


def guard_dtype_size(tensor_arg, arg_name, dtype=None, size=None):
    if dtype is not None and tensor_arg.dtype != dtype:
        raise ValueError(f"Expected Tensor argument {arg_name} to have dtype {dtype}, but got {tensor_arg.dtype} instead.")
    if size is not None and tensor_arg.size() != size:
        raise ValueError(f"Expected Tensor argument {arg_name} to have size {size}, but got {tensor_arg.size()} instead.")

# taken from
# https://github.com/mit-han-lab/smoothquant/blob/2f87951dacfb9238d8d657f52ae83a82a3c9ba0c/smoothquant/fake_quant.py#L26
# and slightly modified
def quantize_activation_per_token_absmax(t):
    # if the shape of t is [B, N, K], the shape of scales will be [B, N, 1]
    mapping_type = MappingType.SYMMETRIC
    block_size = list(t.shape)
    for i in range(len(block_size) - 1):
        block_size[i] = 1
    dtype = torch.int8
    eps = 1e-5
    # Note: the original smoothquant does not clamp to qmin/qmax here,
    # but some of the tests with bfloat16 ended up with a flipped sign
    # if we don't clamp.  TODO(future) look into this further.
    quant_min = -127
    quant_max = 127
    scale_dtype = torch.float32 if t.dtype == torch.float16 else None

    scale, zero_point = choose_qparams_affine(t, mapping_type, block_size, dtype, quant_min, quant_max, eps, scale_dtype=scale_dtype)

    quantized = quantize_affine(t, block_size, scale, zero_point, dtype, quant_min, quant_max)

    return quantized, scale

def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    """
    assumes symmetric quantization
    assumes axis == 0
    assumes dense memory format
    TODO(future): relax ^ as needed
    """

    assert x.dim() == 2, "only support 2d Tensors"

    eps = torch.finfo(torch.float32).eps
    block_size = (1, x.shape[1])
    zero_point_dtype = torch.int64

    mapping_type = MappingType.SYMMETRIC
    scale, zero_point = choose_qparams_affine(x, mapping_type, block_size, target_dtype=target_dtype, quant_min=quant_min, quant_max=quant_max, eps=eps, zero_point_dtype=zero_point_dtype)
    quant = quantize_affine(x, block_size, scale, zero_point, target_dtype, quant_min, quant_max)
    return quant, scale, zero_point

# reference: https://fburl.com/code/vfsygwd0
def dequantize_per_tensor(int_repr, scale, zero_point, out_dtype=torch.float32):
    block_size = int_repr.shape
    input_dtype = int_repr.dtype
    assert scale.numel() == 1, f"scale size: {scale.numel()}"
    dequantized = dequantize_affine(int_repr, block_size, scale, zero_point, input_dtype, output_dtype=out_dtype)
    return dequantized


# reference: https://fburl.com/code/org0fmi3
def dequantize_per_channel(int_repr, scales, zero_points, out_dtype=torch.float32):
    assert int_repr.dim() == 2, "only support 2d Tensors"
    # channel axis == 0
    # block_size before transpose should be (1, int_repr.shape[1]) for axis == 0 per channel quant

    # TODO: transpose is for perf reasons for torch.compile, we should separate this to lowering step
    int_repr = int_repr.t()
    # transpose for block_size as well
    block_size = (int_repr.shape[0], 1)
    input_dtype = int_repr.dtype
    dequantized = dequantize_affine(int_repr, block_size, scales, zero_points, input_dtype, output_dtype=out_dtype)
    dequantized = dequantized.t()
    return dequantized

def get_groupwise_affine_qparams(w, n_bit=4, groupsize=128, dtype=torch.bfloat16):
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2
    assert n_bit <= 8, f"only n_bit smaller than 8 is supported, got: {n_bit}"

    mapping_type = MappingType.ASYMMETRIC
    target_dtype = torch.int32
    block_size = (1, groupsize)
    quant_min = 0
    quant_max = 2**n_bit - 1
    eps = 1e-6
    scale_dtype = dtype
    zero_point_dtype = dtype

    scale, zero_point = choose_qparams_affine(
        w,
        mapping_type,
        block_size,
        target_dtype,
        quant_min,
        quant_max,
        eps,
        scale_dtype=scale_dtype,
        zero_point_dtype=zero_point_dtype,
        preserve_zero=False,
        zero_point_domain=ZeroPointDomain.FLOAT
    )

    return scale.to(dtype=dtype).reshape(w.shape[0], -1), zero_point.to(
        dtype=dtype
    ).reshape(w.shape[0], -1)


def pack_tinygemm_scales_and_zeros(scales, zeros, dtype=torch.bfloat16):
    guard_dtype_size(scales, "scales", dtype=dtype, size=zeros.size())
    guard_dtype_size(zeros, "zeros", dtype=dtype)
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )


def unpack_tinygemm_scales_and_zeros(scales_and_zeros):
    assert len(scales_and_zeros.shape) == 3 and scales_and_zeros.shape[2] == 2
    return torch.split(scales_and_zeros.transpose(0, 1), 1, 2)


def groupwise_affine_quantize_tensor_from_qparams(
    w,
    scales,
    zeros,
    n_bit=4,
    groupsize=128,
):
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    block_size = (1, groupsize)
    output_dtype = torch.int32
    quant_min = 0
    quant_max = 2 ** n_bit - 1

    int_data = quantize_affine(w, block_size, scales, zeros, output_dtype, quant_min, quant_max, zero_point_domain = ZeroPointDomain.FLOAT)
    return int_data

def groupwise_affine_dequantize_tensor_from_qparams(
    w_int4x8,
    scales,
    zeros,
    n_bit=4,
    groupsize=128,
):
    assert groupsize > 1
    # needed for GPTQ single column dequantize
    if groupsize > w_int4x8.shape[-1] and scales.shape[-1] == 1:
        groupsize = w_int4x8.shape[-1]
    assert w_int4x8.shape[-1] % groupsize == 0
    assert w_int4x8.dim() == 2

    block_size = (1, groupsize)
    input_dtype = torch.int32
    quant_min = 0
    quant_max = 2**n_bit - 1
    return dequantize_affine(w_int4x8, block_size, scales, zeros, input_dtype, quant_min, quant_max, zero_point_domain=ZeroPointDomain.FLOAT, output_dtype=scales.dtype)


def groupwise_affine_quantize_tensor(w, n_bit=4, groupsize=128, dtype=torch.bfloat16):
    scales, zeros = get_groupwise_affine_qparams(w, n_bit, groupsize, dtype)
    w_int4x8 = groupwise_affine_quantize_tensor_from_qparams(
        w, scales, zeros, n_bit, groupsize
    )
    scales_and_zeros = pack_tinygemm_scales_and_zeros(scales, zeros, dtype)
    return w_int4x8, scales_and_zeros


def groupwise_affine_dequantize_tensor(
    w_int4x8,
    scales_and_zeros,
    n_bit=4,
    groupsize=128,
):
    scales, zeros = unpack_tinygemm_scales_and_zeros(scales_and_zeros)
    return groupwise_affine_dequantize_tensor_from_qparams(
        w_int4x8, scales, zeros, n_bit, groupsize
    )

def recommended_inductor_config_setter():
    """
    Set inductor config to use the following optimizations which have been showed to improve performance for quantized models:
        coordinate_descent_tuning = True
        coordinate_descent_check_all_directions = True
        force_fuse_int_mm_with_mul = True
        fx_graph_cache = True
        triton.unique_kernel_names = True
        torch.set_float32_matmul_precision("high")
    """
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.coordinate_descent_check_all_directions = True
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.fx_graph_cache = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch.set_float32_matmul_precision("high")

def _get_per_token_block_size(x: torch.Tensor) -> List[int]:
    block_size = []
    for i in range(len(x.shape)-1):
        block_size.append(1)
    block_size.append(x.shape[-1])
    return block_size