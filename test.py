import torch
import argparse
import torchvision.models
from torch2trt import torch2trt
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()
    
    input = torch.randn((1, 3, 224, 224)).cuda().half()
    
    with torch.no_grad():
        model = getattr(torchvision.models, str(args.model))(pretrained=True).cuda().half().eval()
        model_trt = torch2trt(model, [input], fp16_mode=True)

        # run pytorch
        output = model(input)
        t0 = time.time()
        for i in range(100):
            output = model(input)
        t1 = time.time()

        dt_pytorch = (t1 - t0) / 100.0

        output = model_trt(input)
        t0 = time.time()
        for i in range(100):
            output = model_trt(input)
        t1 = time.time()

        dt_tensorrt = (t1 - t0) / 100.0

        line = '%s\t%f\t%f' % (args.model, dt_pytorch, dt_tensorrt)

        print(line)

        with open('timings.txt', 'a') as f:
            f.write(line + '\n')