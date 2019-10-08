from torch2trt import *
import torchvision
import torch
from .segmentation import deeplabv3_resnet50


if __name__ == '__main__':
    model = deeplabv3_resnet50().cuda().eval().half()
    data = torch.randn((1, 3, 224, 224)).cuda().half()
    
    print('Running torch2trt...')
    model_trt = torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)

    print('Saving model...')
    torch.save(model_trt.state_dict(), '.test_model.pth')

    print('Loading model...')
    model_trt_2 = TRTModule()
    model_trt_2.load_state_dict(torch.load('.test_model.pth'))

    assert(model_trt_2.engine is not None)
    
    print(torch.max(torch.abs(model_trt_2(data) - model(data))))
    print(torch.max(torch.abs(model_trt_2(data) - model_trt(data))))