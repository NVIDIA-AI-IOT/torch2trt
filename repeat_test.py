import torch
class Foo(torch.nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.arg=arg
    def forward(self, x):
        return x.repeat(self.arg)

if __name__ == "__main__":
    model = Foo((10)).eval().cuda()
    x = torch.ones([1], device='cuda')
    y = model(x)

    from torch2trt import torch2trt
    model_trt = torch2trt(model, [x])
    y_trt = model_trt(x)
    print(y.size())
    print(y)
    print(y_trt.size())
    print(y_trt)
