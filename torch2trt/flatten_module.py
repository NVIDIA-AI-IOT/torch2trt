import torch
import torch.nn as nn
from .flattener import Flattener


class Unflatten(nn.Module):

    def __init__(self, module, input_flattener=None, output_flattener=None):
        super().__init__()
        self.module = module
        self.input_flattener = input_flattener
        self.output_flattener = output_flattener

    def forward(self, *args):
        if self.input_flattener is not None:
            args = self.input_flattener.flatten(args)
        output = self.module(*args)
        if self.output_flattener is not None:
            output = self.output_flattener.unflatten(output)
        return output


class Flatten(nn.Module):

    def __init__(self, module, input_flattener=None, output_flattener=None):
        super().__init__()
        self.module = module
        self.input_flattener = input_flattener
        self.output_flattener = output_flattener

    def forward(self, *args):
        if self.input_flattener is not None:
            args = self.input_flattener.unflatten(*args)
        output = self.module(*args)
        if self.output_flattener is not None:
            output = self.output_flattener.flatten(output)
        return output