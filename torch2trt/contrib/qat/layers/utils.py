"""
Utility file for quantization ops
"""
import torch

# Custom flag to export trt
class HelperFunction(object):
    """
    Helper class used to export TRT model
    """
    export_trt = False
    def __init__():
        super().__init__()


class InferQuantTensor(torch.nn.Module):
    """
    Class to maintain quantization parameters during inference and trt conversion
    Parameters that will be tracked:
    amax : maximum possible value in a tensor
    scale : quantization scale derived from amax
    zero_point: zero_point of a tensor used for trt conversion
    quant_min: minimum value of INT8 range
    quant_max : maximum value of INT8 range
    axis : axis along which quantization will occur (used in per channel quantization)
    """
    export_trt = False

    def __init__(self):
        super().__init__()
        ## All these values will be initialized later on.
        amax = None
        scale = None
        zero_point = None
        quant_min = None
        quant_max = None

    def _extract_info(self,amax, _num_bits, _unsigned):
        bound = (1 << (_num_bits - 1 + int(_unsigned))) - 1
        if amax.numel() == 1:
            scale=amax.item() / bound
            zero_point = 0
            quant_min = -bound - 1 if not _unsigned else 0
            quant_max = bound
            axis = None
        else:
            amax_sequeeze = amax.squeeze()
            if len(amax_sequeeze.shape) != 1:
                raise TypeError("Multiple axis is not supported in quantization")
            quant_dim = list(amax.shape).index(list(amax_sequeeze.shape)[0])
            scale = amax_sequeeze / bound
            scale = scale.data
            zero_point = torch.zeros_like(scale, dtype=torch.int32).data
            axis = quant_dim
            quant_min = -bound - 1 if not _unsigned else 0
            quant_max = bound
        
        amax = self.correct_tensor_type(amax)
        scale = self.correct_tensor_type(scale)
        zero_point = self.correct_tensor_type(zero_point)
        quant_min = self.correct_tensor_type(quant_min)
        quant_max = self.correct_tensor_type(quant_max)
        axis = self.correct_tensor_type(axis)
        return amax, scale, zero_point, quant_min, quant_max, axis

    def correct_tensor_type(self,variable):
        if torch.is_tensor(variable):
            return torch.nn.Parameter(variable,requires_grad=False)
        elif variable is None:
            return variable
        else:
            return torch.nn.Parameter(torch.as_tensor([variable]),requires_grad=False)

    def extract_quant_info(self, amax, num_bits, unsigned):

        amax, scale, zero_point,quant_min, quant_max, axis = self._extract_info(amax, num_bits, unsigned)
        
        self.amax = amax
        self.quant_scale = scale
        self.zero_point = zero_point
        self.quant_min = quant_min
        self.quant_max = quant_max
        if not axis == None:
            self.quant_axis = axis

    def quantize_tensor(self,input):
        if self.amax.numel() == 1:
            quant_input = torch.fake_quantize_per_tensor_affine(input,
                    self.quant_scale.to(torch.float32).item(),
                    self.zero_point.to(torch.long).item(),
                    self.quant_min.to(torch.long).item(),
                    self.quant_max.to(torch.long).item())
        else:
            if torch.__version__ > "1.9":
                zero_pt = self.zero_point.to(torch.int32)
            else:
                zero_pt = self.zero_point.to(torch.long)

            quant_input = torch.fake_quantize_per_channel_affine(input,
                    self.quant_scale,
                    zero_pt,
                    self.quant_axis.to(torch.int32).item(),
                    self.quant_min.to(torch.long).item(),
                    self.quant_max.to(torch.long).item())

        return quant_input

    def forward(self, input):
        return self.quantize_tensor(input)
