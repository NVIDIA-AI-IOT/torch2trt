from argparse import ArgumentParser
from torch2trt.dataset import FolderDataset, ListDataset
from torch2trt import torch2trt, TRTModule
from easyocr import Reader
import tensorrt as trt
import torch
import time
from tempfile import mkdtemp

parser = ArgumentParser()
parser.add_argument('--detector_data', type=str, default='detector_data')
parser.add_argument('--output', type=str, default='detector_trt.pth')
parser.add_argument('--int8', action='store_true')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--dla', action='store_true')
parser.add_argument('--dla_core', type=int, default=0)
args = parser.parse_args()

detector_dataset = FolderDataset(args.detector_data)

if len(detector_dataset) == 0:
    raise ValueError('Detector dataset is empty, make sure to run generate_data.py first.')

reader = Reader(['en'])
detector_torch = reader.detector.module

if args.int8:
    num_calib = 5
    calib_dataset = FolderDataset(mkdtemp())
    for i in range(num_calib):
        calib_dataset.insert(tuple([t + 0.2 * torch.randn_like(t) for t in detector_dataset[i % len(detector_dataset)]]))

print('Running torch2trt...')
detector_trt = torch2trt(
    detector_torch,
    detector_dataset,
    int8_mode=args.int8,
    fp16_mode=args.fp16,
    default_device_type=trt.DeviceType.DLA if args.dla else trt.DeviceType.GPU,
    max_workspace_size=1 << 26,
    log_level=trt.Logger.VERBOSE,
    int8_calib_dataset=calib_dataset if args.int8 else None,
    int8_calib_algorithm=trt.CalibrationAlgoType.MINMAX_CALIBRATION,
    use_onnx=True
)

torch.save(detector_trt.state_dict(), args.output)

def profile_module(module, dataset, count=None):
    
    if count is None:
        count = len(dataset)

    output = module(*dataset[0]) # warmup

    torch.cuda.current_stream().synchronize()
    t0 = time.monotonic()
    for i in range(count):
        output = module(*dataset[i % len(dataset)])
    torch.cuda.current_stream().synchronize()
    t1 = time.monotonic()

    return count / (t1 - t0)

print('Profiling PyTorch...')
fps_torch = profile_module(detector_torch, detector_dataset, 30)
print(f'FPS Torch: {fps_torch}')

print('Profiling TensorRT')
fps_trt = profile_module(detector_trt, detector_dataset, 30)
print(f'FPS TensorRT: {fps_trt}')