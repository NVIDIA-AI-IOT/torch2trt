import torch
from torch.nn.intrinsic.qat.modules.conv_fused import ConvBnReLU2d as CBR
from torch.nn.intrinsic.qat.modules.conv_fused import ConvReLU2d as CR

from torch2trt import torch2trt

torch.manual_seed(58394)
qconfig=torch.quantization.QConfig(activation=torch.quantization.FakeQuantize.with_args(observer=torch.quantization.MovingAverageMinMaxObserver,dtype=torch.qint8,reduce_range=False,qscheme=torch.per_tensor_symmetric,quant_min=-128,quant_max=127),weight=torch.quantization.FakeQuantize.with_args(observer=torch.quantization.MovingAverageMinMaxObserver,dtype=torch.qint8,qscheme=torch.per_tensor_symmetric,quant_min=-127,quant_max=127))

class vanilla_cnn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.layer1 = CBR(1,1,3,padding=1,qconfig=qconfig)
        self.layer1 = CR(1,1,3,padding=1,qconfig=qconfig)
        self.layer1.weight_fake_quant.activation_post_process.min_val=torch.tensor([-2],dtype=torch.float32)
        self.layer1.weight_fake_quant.activation_post_process.max_val=torch.tensor([2],dtype=torch.float32)
        self.layer1.activation_post_process.activation_post_process.min_val=torch.tensor([-9],dtype=torch.float32)
        self.layer1.activation_post_process.activation_post_process.max_val=torch.tensor([9],dtype=torch.float32)
        self.layer1.weight = torch.nn.Parameter(torch.ones([1,1,3,3]))

    def forward(self,inputs):
        return self.layer1(inputs)


model = vanilla_cnn().eval()
print(model.layer1.weight.size())
print("-------------")
print(model)
print("-------------")

#input_t = torch.randint(32,[1,1,6,6],dtype=torch.float32)
input_t = torch.ones([1,1,6,6],dtype=torch.float32)
#converted_model = torch.quantization.convert(model)
model_out = model(input_t)
#cmodel_out = converted_model(input_t)
print("---------------------")
print(input_t)
print(">>>> Model output")
print(model_out)
print("---------------------")
#print(">>>> Converted Model output")
#print(cmodel_out)

model = model.eval().cuda()
input_t = input_t.to("cuda:0")

trt_model = torch2trt(model,[input_t],int8_mode=True)

print("MODEL CONVERSION COMPLETE")
trt_out = trt_model(input_t)

print(trt_out)
