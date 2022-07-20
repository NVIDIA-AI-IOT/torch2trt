from argparse import ArgumentParser
from torch2trt.dataset import FolderDataset
from torch2trt import torch2trt, TRTModule
from easyocr import Reader
import tensorrt as trt
import torch
import time
from tempfile import mkdtemp


parser = ArgumentParser()
parser.add_argument('--detector_data', type=str, default='detector_data')
parser.add_argument('--recognizer_data', type=str, default='recognizer_data')
parser.add_argument('--output', type=str, default='recognizer_trt.pth')
parser.add_argument('--int8', action='store_true')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--max_workspace_size', type=int, default=1<<28)
args = parser.parse_args()

detector_dataset = FolderDataset(args.detector_data)
recognizer_dataset = FolderDataset(args.recognizer_data)

if len(detector_dataset) == 0:
    raise ValueError('Detector dataset is empty, make sure to run generate_data.py first.')

if len(recognizer_dataset) == 0:
    raise ValueError('Recognizer dataset is empty, make sure to run generate_data.py first.')


if args.int8:
    num_calib = 200
    calib_dataset = FolderDataset(mkdtemp())
    for i in range(num_calib):
        calib_dataset.insert(tuple([t.float() + 0.2 * torch.randn_like(t.float()) for t in recognizer_dataset[i % len(recognizer_dataset)]]))

reader = Reader(['en'])
module_torch = reader.detector.module

max_shapes = list(recognizer_dataset.max_shapes())

# override default max shape to use full image width
max_shapes[0] = torch.Size((
    recognizer_dataset.max_shapes()[0][0],
    recognizer_dataset.max_shapes()[0][1],
    recognizer_dataset.max_shapes()[0][2],
    detector_dataset.max_shapes()[0][3]
))
max_shapes = tuple(max_shapes)

class PoolFix(torch.nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=-1, keepdim=True)

if isinstance(reader.recognizer.module.AdaptiveAvgPool, torch.nn.AdaptiveAvgPool2d):
    reader.recognizer.module.AdaptiveAvgPool = PoolFix()

recognizer_torch = reader.recognizer.module

print('Running torch2trt...')
recognizer_trt = torch2trt(
    reader.recognizer.module, 
    recognizer_dataset, 
    max_shapes=max_shapes, 
    use_onnx=True,  # LSTM currently only implemented in ONNX workflow
    fp16_mode=args.fp16,
    int8_mode=args.int8,
    max_workspace_size=args.max_workspace_size,
    log_level=trt.Logger.VERBOSE
)

# recognizer_trt.ignore_inputs = [1]

torch.save(recognizer_trt.state_dict(), args.output)

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
fps_torch = profile_module(recognizer_torch, recognizer_dataset, 50)
print(f'FPS Torch: {fps_torch}')

print('Profiling TensorRT')
fps_trt = profile_module(recognizer_trt, recognizer_dataset, 30)
print(f'FPS TensorRT: {fps_trt}')