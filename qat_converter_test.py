import torch
from torch.nn.intrinsic.qat.modules.conv_fused import ConvBnReLU2d as CBR
from torch2trt import torch2trt

torch.manual_seed(54321)
qconfig=torch.quantization.QConfig(activation=torch.quantization.FakeQuantize.with_args(observer=torch.quantization.MovingAverageMinMaxObserver,dtype=torch.quint8,reduce_range=False,qscheme=torch.per_tensor_symmetric,quant_min=0,quant_max=255),weight=torch.quantization.FakeQuantize.with_args(observer=torch.quantization.MovingAverageMinMaxObserver,dtype=torch.qint8,qscheme=torch.per_tensor_symmetric,quant_min=-127,quant_max=127))

class vanilla_cnn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = CBR(1,1,3,qconfig=qconfig)
        self.layer1.weight_fake_quant.activation_post_process.min_val=torch.tensor([-127],dtype=torch.float32)
        self.layer1.weight_fake_quant.activation_post_process.max_val=torch.tensor([127],dtype=torch.float32)
        self.layer1.activation_post_process.activation_post_process.min_val=torch.tensor([0],dtype=torch.float32)
        self.layer1.activation_post_process.activation_post_process.max_val=torch.tensor([255],dtype=torch.float32)

    def forward(self,inputs):
        return self.layer1(inputs)


model = vanilla_cnn().eval().cuda()
print("-------------")
print(model)
print("-------------")

input_t = torch.randn([1,1,6,6],dtype=torch.float32,device="cuda:0")

model_out = model(input_t)

print("---------------------")
print(input_t)
print(">>>>")
print(model_out)
print("---------------------")

trt_model = torch2trt(model,[input_t],int8_mode=True)

print("MODEL CONVERSION COMPLETE")
trt_out = trt_model(input_t)

print(trt_out)
