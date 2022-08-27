import torch
from torch2trt import torch2trt


def test_contiguous():

    torch.manual_seed(0)

    net = torch.nn.Conv2d(3, 10, kernel_size=3)
    net.eval().cuda()

    test_tensor = torch.randn((1, 25, 25, 3)).cuda().permute((0, 3, 1, 2))

    with torch.no_grad():
        test_out = net(test_tensor)

    with torch.no_grad():
        trt_net = torch2trt(net, [test_tensor])
        test_trt_out = trt_net(test_tensor)

    delta = torch.max((test_out.contiguous() - test_trt_out.contiguous()).abs())
    assert delta < 1e-3, f"Delta: {delta}"

