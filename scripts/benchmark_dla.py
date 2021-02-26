 # for convenience, use local not installed torch2trt
import sys
sys.path.append('..') 

# other imports
import argparse
import os
import torch
import torchvision
import time
import torch.distributed as dist  # used to test concurrent model execution
from torch2trt import torch2trt, trt, device_type_str, TRTModule


def benchmark(data, model, iterations, nvvp_output_path, use_nvvp):

    output = model(data)
    torch.cuda.current_stream().synchronize()

    t0 = time.monotonic()
    torch.cuda.current_stream().synchronize()
    for i in range(iterations):
        output = model(data)
    torch.cuda.current_stream().synchronize()
    t1 = time.monotonic()

    if use_nvvp:
        torch.cuda.profiler.init(nvvp_output_path, output_mode='csv')
        torch.cuda.current_stream().synchronize()
        with torch.cuda.profiler.profile():
            output = model(data)
            torch.cuda.current_stream().synchronize()

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
    elif name == 'resnet50_fp16_gpu':
        model = torchvision.models.resnet50(pretrained=True)
        torch2trt_kwargs = {
            'fp16_mode': True,
            'default_device_type': trt.DeviceType.GPU
        }
    elif name == 'resnet50_int8_gpu':
        model = torchvision.models.resnet50(pretrained=True)
        torch2trt_kwargs = {
            'int8_mode': True,
            'default_device_type': trt.DeviceType.GPU
        }
    elif name == 'resnet50_fp16_dla':
        model = torchvision.models.resnet50(pretrained=True)
        torch2trt_kwargs = {
            'fp16_mode': True,
            'default_device_type': trt.DeviceType.DLA
        }
    elif name == 'resnet50_int8_dla':
        model = torchvision.models.resnet50(pretrained=True)
        torch2trt_kwargs = {
            'int8_mode': True,
            'default_device_type': trt.DeviceType.DLA
        }
    elif name == 'resnet50_fp16_gpu_dla1':
        model = torchvision.models.resnet50(pretrained=True)
        torch2trt_kwargs = {
            'fp16_mode': True,
            'default_device_type': trt.DeviceType.GPU,
            'device_types': {
                model.layer1: trt.DeviceType.DLA
            }
        }
    elif name == 'resnet50_fp16_gpu_dla12':
        model = torchvision.models.resnet50(pretrained=True)
        torch2trt_kwargs = {
            'fp16_mode': True,
            'default_device_type': trt.DeviceType.GPU,
            'device_types': {
                model.layer1: trt.DeviceType.DLA,
                model.layer2: trt.DeviceType.DLA
            }
        }
    elif name == 'resnet50_fp16_gpu_dla123':
        model = torchvision.models.resnet50(pretrained=True)
        torch2trt_kwargs = {
            'fp16_mode': True,
            'default_device_type': trt.DeviceType.GPU,
            'device_types': {
                model.layer1: trt.DeviceType.DLA,
                model.layer2: trt.DeviceType.DLA,
                model.layer3: trt.DeviceType.DLA
            }
        }
    elif name == 'resnet50_int8_gpu_dla1':
        model = torchvision.models.resnet50(pretrained=True)
        torch2trt_kwargs = {
            'int8_mode': True,
            'default_device_type': trt.DeviceType.GPU,
            'device_types': {
                model.layer1: trt.DeviceType.DLA
            }
        }
    elif name == 'resnet50_int8_gpu_dla12':
        model = torchvision.models.resnet50(pretrained=True)
        torch2trt_kwargs = {
            'int8_mode': True,
            'default_device_type': trt.DeviceType.GPU,
            'device_types': {
                model.layer1: trt.DeviceType.DLA,
                model.layer2: trt.DeviceType.DLA
            }
        }
    elif name == 'resnet50_int8_gpu_dla123':
        model = torchvision.models.resnet50(pretrained=True)
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
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--model_cache_dir', type=str, default='model_cache')
    parser.add_argument('--nvvp_output_dir', type=str, default='nvvp')
    parser.add_argument('--nvvp', action='store_true')
    parser.add_argument('--dla_core', type=int, default=0)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()

    if args.use_cache and not os.path.exists(args.model_cache_dir):
        os.makedirs(args.model_cache_dir)

    if args.nvvp and not os.path.exists(args.nvvp_output_dir):
        os.makedirs(args.nvvp_output_dir)
    
    torch_nvvp_path = os.path.join(args.nvvp_output_dir, args.model + '_torch.nvvp')
    trt_nvvp_path = os.path.join(args.nvvp_output_dir, args.model + '_trt.nvvp')
    model_path = os.path.join(args.model_cache_dir, args.model + '_trt_bs{bs}_dlacore{dla}.pth'.format(bs=args.batch_size, dla=args.dla_core))

    model, kwargs = get_benchmark_config(args.model)

    kwargs.update({
        'max_batch_size': args.batch_size,
        'dla_core': args.dla_core
    })

    if args.distributed:
        dist.init_process_group(
            backend='gloo',
            init_method='file:///distributed_test',
            world_size=args.world_size,
            rank=args.rank
        )

    model = model.cuda().eval()
    data = torch.randn(args.batch_size, 3, 224, 224).cuda()

    if args.use_cache and os.path.exists(model_path):
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(model_path))
    else:
        model_trt = torch2trt(model, [data], **kwargs)
        if args.use_cache:
            torch.save(model_trt.state_dict(), model_path)


    fps_torch = benchmark(data, model, args.benchmark_iters, torch_nvvp_path, args.nvvp) * args.batch_size

    # wait for processes to join if we're running concurrent models
    if args.distributed:
        dist.barrier()
        print('joined process %d' % args.rank)

    fps_trt = benchmark(data, model_trt, args.benchmark_iters, trt_nvvp_path, args.nvvp) * args.batch_size

    if 'device_types' in kwargs:
        kwargs.update({'device_types': device_types_string(model, kwargs['device_types'])})

    print('| {name} | {kwargs} | {fps_torch} | {fps_trt} |'.format(
        name=args.model,
        kwargs=kwargs,
        fps_torch=fps_torch,
        fps_trt=fps_trt
    ))
    