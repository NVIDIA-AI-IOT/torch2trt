import torch
class Foo(torch.nn.Module):
    def __init__(self,dim,start,length):
        super().__init__()
        self.start = start
        self.dim=dim
        self.length=length
    def forward(self, x):
        return torch.narrow(x,self.dim,self.start,self.length)

if __name__ == "__main__":
    model = Foo(2,0,50).eval().cuda()
    x = torch.randn([1,3,224,224], device='cuda')
    y = model(x)

    from torch2trt import torch2trt
    model_trt = torch2trt(model, [x])
    y_trt = model_trt(x)
    print(y.size())
    print(y_trt.size())
