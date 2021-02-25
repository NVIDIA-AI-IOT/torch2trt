 # for convenience, use local not installed torch2trt
import sys
sys.path.append('..') 

# other imports
import argparse
import torch
import torchvision
import time
from torch2trt import torch2trt, trt, device_type_str


def benchmark(data, model, iterations):
    output = model(data)
    torch.cuda.current_stream().synchronize()

    t0 = time.monotonic()
    torch.cuda.current_stream().synchronize()
    for i in range(iterations):
        output = model(data)
    torch.cuda.current_stream().synchronize()
    t1 = time.monotonic()
    return float(iterations / (t1 - t0))


def get_benchmark_config(name):
    if name == 'resnet18_fp16_gpu':
        model = torchvision.models.resnet18(pretrained=True)
        torch2trt_kwargs = {
            'fp16_mode': True,
            'default_device_type': trt.DeviceType.GPU
        }
    elif name == 'resnet18_int8_gpu':
        model = torchvision.models.resnet18(pretrained=True)
        torch2trt_kwargs = {
            'int8_mode': True,
            'default_device_type': trt.DeviceType.GPU
        }
    elif name == 'resnet18_fp16_dla':
        model = torchvision.models.resnet18(pretrained=True)
        torch2trt_kwargs = {
            'fp16_mode': True,
            'default_device_type': trt.DeviceType.DLA
        }
    elif name == 'resnet18_int8_dla':
        model = torchvision.models.resnet18(pretrained=True)
        torch2trt_kwargs = {
            'int8_mode': True,
            'default_device_type': trt.DeviceType.DLA
        }
    elif name == 'resnet18_fp16_gpu_dla1':
        model = torchvision.models.resnet18(pretrained=True)
        torch2trt_kwargs = {
            'fp16_mode': True,
            'default_device_type': trt.DeviceType.GPU,
            'device_types': {
                model.layer1: trt.DeviceType.DLA
            }
        }
    elif name == 'resnet18_fp16_gpu_dla12':
        model = torchvision.models.resnet18(pretrained=True)
        torch2trt_kwargs = {
            'fp16_mode': True,
            'default_device_type': trt.DeviceType.GPU,
            'device_types': {
                model.layer1: trt.DeviceType.DLA,
                model.layer2: trt.DeviceType.DLA
            }
        }
    elif name == 'resnet18_fp16_gpu_dla123':
        model = torchvision.models.resnet18(pretrained=True)
        torch2trt_kwargs = {
            'fp16_mode': True,
            'default_device_type': trt.DeviceType.GPU,
            'device_types': {
                model.layer1: trt.DeviceType.DLA,
                model.layer2: trt.DeviceType.DLA,
                model.layer3: trt.DeviceType.DLA
            }
        }
    elif name == 'resnet18_int8_gpu_dla1':
        model = torchvision.models.resnet18(pretrained=True)
        torch2trt_kwargs = {
            'int8_mode': True,
            'default_device_type': trt.DeviceType.GPU,
            'device_types': {
                model.layer1: trt.DeviceType.DLA
            }
        }
    elif name == 'resnet18_int8_gpu_dla12':
        model = torchvision.models.resnet18(pretrained=True)
        torch2trt_kwargs = {
            'int8_mode': True,
            'default_device_type': trt.DeviceType.GPU,
            'device_types': {
                model.layer1: trt.DeviceType.DLA,
                model.layer2: trt.DeviceType.DLA
            }
        }
    elif name == 'resnet18_int8_gpu_dla123':
        model = torchvision.models.resnet18(pretrained=True)
        torch2trt_kwargs = {
            'int8_mode': True,
            'default_device_type': trt.DeviceType.GPU,
            'device_types': {
                model.layer1: trt.DeviceType.DLA,
                model.layer2: trt.DeviceType.DLA,
                model.layer3: trt.DeviceType.DLA
            }
        }
    else:
        raise RuntimeError('Module configuration is not recognized.')

    
    return model, torch2trt_kwargs

def device_types_string(model, device_types):
    str_repr = {}
    for module_a, device_type in device_types.items():
        for name, module_b in model.named_modules():
            if module_b == module_a:
                str_repr[name] = device_type_str(device_type)
    return str(str_repr)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18_fp16_gpu')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--benchmark_iters', type=float, default=50)
    args = parser.parse_args()

    model, kwargs = get_benchmark_config(args.model)

    kwargs.update({
        'max_batch_size': args.batch_size
    })


    model = model.cuda().eval()
    data = torch.randn(args.batch_size, 3, 224, 224).cuda()


    model_trt = torch2trt(model, [data], **kwargs)

    fps_torch = benchmark(data, model, args.benchmark_iters) * args.batch_size
    fps_trt = benchmark(data, model_trt, args.benchmark_iters) * args.batch_size

    if 'device_types' in kwargs:
        kwargs.update({'device_types': device_types_string(model, kwargs['device_types'])})

    print('| {name} | {kwargs} | {fps_torch} | {fps_trt} |'.format(
        name=args.model,
        kwargs=kwargs,
        fps_torch=fps_torch,
        fps_trt=fps_trt
    ))