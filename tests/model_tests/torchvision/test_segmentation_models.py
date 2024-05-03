import torch
import torchvision
import torch2trt


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)['out']


def _cross_validate_module(model, shape=(224, 224)):
    model = model.cuda().eval()
    data = torch.randn(1, 3, *shape).cuda()
    model_trt = torch2trt.torch2trt(model, [data])
    data = torch.randn(1, 3, *shape).cuda()
    out = model(data)
    out_trt = model_trt(data)
    assert torch.allclose(out, out_trt, rtol=1e-2, atol=1e-2)



def test_deeplabv3_resnet50():
    bb = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
    model = ModelWrapper(bb)
    _cross_validate_module(model)


def test_deeplabv3_resnet101():
    bb = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False)
    model = ModelWrapper(bb)
    _cross_validate_module(model)


def test_fcn_resnet50():
    bb = torchvision.models.segmentation.fcn_resnet50(pretrained=False)
    model = ModelWrapper(bb)
    _cross_validate_module(model)


def test_fcn_resnet101():
    bb = torchvision.models.segmentation.fcn_resnet101(pretrained=False)
    model = ModelWrapper(bb)
    _cross_validate_module(model)