import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from GPTQQuantizer import Int4WeightOnlyGPTQQuantizer
from InputRecorder import InputRecorder

torch._dynamo.reset()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

precision = torch.bfloat16
device = "cuda"

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def prepare_inputs_for_model(inps, max_new_tokens=1):
    # this is because input from lm-eval is 2d
    if inps.dim() > 2:
        raise ValueError(f"Expected input to be of dim 1 or 2, but got {inps.dim()}")

    input_pos = torch.arange(0, inps.numel(), device=inps.device)
    return (inps.view(1, -1), input_pos)

blocksize = 128
percdamp = 0.01
groupsize = 128
calibration_tasks = ["wikitext"]
calibration_limit = 1
calibration_seq_length = 100
input_prep_func = prepare_inputs_for_model
pad_calibration_inputs = False

inputs = InputRecorder(
    tokenizer,
    calibration_seq_length,
    input_prep_func,
    pad_calibration_inputs,
    model.config.vocab_size,
    device="cpu",
).record_inputs(
    calibration_tasks,
    calibration_limit,
).get_inputs()

quantizer = Int4WeightOnlyGPTQQuantizer(
    blocksize,
    percdamp,
    groupsize,
)

model = quantizer.quantize(model, inputs).cuda()