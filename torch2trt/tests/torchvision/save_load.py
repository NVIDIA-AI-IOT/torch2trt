
from torch2trt.torch2trt import torch2trt, TRTModule


if __name__ == '__main__':
    import torchvision
    import torch
    model = torchvision.models.resnet18(pretrained=True).cuda().eval()
    data = torch.randn((1, 3, 224, 224)).cuda()

    print('Running torch2trt...')
    model_trt = torch2trt(model, [data])

    print('Saving model...')
    torch.save(model_trt.state_dict(), '.test_model.pth')

    print('Loading model...')
    model_trt_2 = TRTModule()
    model_trt_2.load_state_dict(torch.load('.test_model.pth'))

    assert(model_trt_2.engine is not None)
    
    print(torch.max(torch.abs(model_trt_2(data) - model(data))))
    print(torch.max(torch.abs(model_trt_2(data) - model_trt(data))))