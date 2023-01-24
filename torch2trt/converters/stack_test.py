import torch
import torch2trt

torch.manual_seed(0)


class StackModule(torch.nn.Module):
    def forward(self, a):
        return torch.stack((a, a))


def test_stack_dynamic():
    tensor = torch.rand((2, 2)).cuda()

    model = StackModule().eval().cuda()
    model_trt = torch2trt.torch2trt(
        model, [tensor], min_shapes=[(1, 1)], max_shapes=[(5, 5)]
    )

    assert torch.allclose(model(tensor), model_trt(tensor))

    tensor = torch.rand((5, 5)).cuda()
    assert torch.allclose(model(tensor), model_trt(tensor))


if __name__ == "__main__":
    test_stack_dynamic()

