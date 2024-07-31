import torch
import torch.nn.functional as F
import lm_eval
from lm_eval.evaluator import evaluate  # pyre-ignore[21]
from lm_eval.models.huggingface import HFLM as eval_wrapper  # pyre-ignore[21]
from lm_eval.tasks import get_task_dict
from MultiInput import _MultiInput

class InputRecorder(eval_wrapper):
    """
    This is a fake evaluation wrapper from the lm_eval library that just records the inputs
    so that they can be used in calibration.

    If pad_calibration_inputs is enabled, the input recorder will take
    each input and pad/truncate it down to the calibration_seq_length.
    (if using padding you should set the embeddings for the pad_token to 0
    in the model)

    Note: after padding/truncation, input_prep_function is called to bring
    it to the proper form to be inserted into a given model.

    If not, it will only truncate inputs to the desired length.
    """

    def __init__(
        self,
        tokenizer,
        calibration_seq_length,
        input_prep_func=None,
        pad_calibration_inputs=False,
        vocab_size=32000,
        pad_token=0,
        device="cpu",
    ):
        try:
            super().__init__()
        except TypeError:
            # lm_eval 0.4.2 removed the default init
            super().__init__("gpt2", device="cpu")

        self.tokenizer = tokenizer
        self._device = torch.device(device)
        self.vocab_size = vocab_size
        self._max_seq_length = calibration_seq_length
        self.calibration_seq_length = calibration_seq_length

        # need to take inps and convert to corrent input
        # for model
        self.input_prep_func = (
            input_prep_func if input_prep_func is not None
            else lambda x: (x,)
        )

        self.pad_calibration_inputs = pad_calibration_inputs
        self.pad_token = pad_token

        self.inputs = None

    @property
    def eot_token_id(self):
        try:
            return self.tokenizer.eos_id()
        except:
            return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 50

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs):
        # TODO: verify this for multi-batch as well
        tokens = self.tokenizer.encode(string)
        if hasattr(self.tokenizer, "bos_id"):
            try:
                tokens = [self.tokenizer.bos_id()] + tokens
            except:
                tokens = [self.tokenizer.bos_id] + tokens
        return tokens

    def tok_decode(self, tokens):
        decoded = self.tokenizer.decode(tokens)
        return decoded

    def add_input(self, args):
        if self.inputs is None:
            self.inputs = [_MultiInput([arg]) for arg in args]
        else:
            self.inputs = [
                multi.add_input(arg) for (multi, arg) in zip(self.inputs, args)
            ]

    def record_inputs(
        self,
        calibration_tasks,
        calibration_limit,
    ):
        try:
            lm_eval.tasks.initialize_tasks()
        except:
            pass

        task_dict = get_task_dict(calibration_tasks)
        print("Obtaining GPTQ calibration inputs on: ", calibration_tasks)

        evaluate(
            self,
            task_dict,
            limit=calibration_limit,
        )
        return self

    def get_inputs(self):
        return self.inputs

    def _model_call(self, inps):
        inps = inps.squeeze(0)
        T = len(inps)
        if (
            # can't use inputs that are too short when padding disabled
            (T < self.calibration_seq_length and not self.pad_calibration_inputs)
            or
            # can't use inputs that actually use token we use for padding
            (self.pad_calibration_inputs and self.pad_token in inps)
        ):
            # give random output
            return torch.randn(
                (1, T, self.vocab_size), dtype=torch.bfloat16, device=self._device
            )

        # pad or truncate to the right size
        if T >= self.calibration_seq_length:
            inps = inps[: self.calibration_seq_length]
        else:
            inps = F.pad(inps, (self.pad_token, self.calibration_seq_length - T))

        inps = inps.unsqueeze(0)
        model_in = self.input_prep_func(inps)

        self.add_input(model_in)

        # output `something` with correct shape to keep eval going
        return torch.randn(
            (1, T, self.vocab_size), dtype=torch.bfloat16, device=self._device
        )

    def _model_generate(self, context, max_length, eos_token_id):
        raise Exception("unimplemented")