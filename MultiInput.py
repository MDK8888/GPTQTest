import torch

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