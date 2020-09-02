import torch
from torch2trt import torch2trt
class Foo(torch.nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.arg=arg
    def forward(self, x):
        return x.repeat(self.arg)

if __name__ == "__main__":
    model = Foo((3,3)).cuda().eval()
    x = [torch.tensor([4],dtype=torch.float32, device='cuda')]
    y = model(*x)

    model_trt = torch2trt(model, x, max_batch_size=1,input_names=['in'],output_names=['out'])
    y_trt = model_trt(*x)
    print(y.size())
    print(y_trt.size())
