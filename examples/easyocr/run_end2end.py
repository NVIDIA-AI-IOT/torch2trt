from argparse import ArgumentParser
import cv2
import torch
import glob
from easyocr import Reader
from torch2trt.dataset import FolderDataset
from torch2trt import torch2trt, TRTModule
import math
import time
import os

parser = ArgumentParser()
parser.add_argument('--images', type=str, default='images')
parser.add_argument('--detector_trt', type=str, default='detector_trt.pth')
parser.add_argument('--recognizer_trt', type=str, default='recognizer_trt.pth')
parser.add_argument('--max_image_area', type=int, default=1280*720)
parser.add_argument('--count', type=int, default=None)
parser.add_argument('--recognizer_batch_size', type=int, default=1)
args = parser.parse_args()


def shrink_to_area(image, area):
    height = image.shape[0]
    width = image.shape[1]

    if height * width > area:
        ar = width / height
        new_height = math.sqrt(area / ar)
        new_width = ar * new_height
        new_height = math.floor(new_height)
        new_width = math.floor(new_width)
        print(f'Resizing {width}x{height} to {new_width}x{new_height}')
        image = cv2.resize(image, (new_width, new_height))

    return image

image_paths = glob.glob(os.path.join(args.images, '*.jpg'))

def profile_reader(reader):

    cumulative_execution_time = 0

    if args.count is None:
        count = len(image_paths)
    else:
        count = args.count

    for i in range(count):

        path = image_paths[i % len(image_paths)]
        image = cv2.imread(path)

        image = shrink_to_area(image, args.max_image_area)
        
        t0 = time.monotonic()
        reader.readtext(image, batch_size=args.recognizer_batch_size)
        t1 = time.monotonic()

        cumulative_execution_time += (t1 - t0)
    
    return count / cumulative_execution_time


reader = Reader(['en'])

detector_trt = TRTModule()
detector_trt.load_state_dict(torch.load(args.detector_trt))

recognizer_trt = TRTModule()
recognizer_trt.load_state_dict(torch.load(args.recognizer_trt))

test_image = shrink_to_area(cv2.imread(image_paths[0]), args.max_image_area)

print('Dumping torch output...')
print(reader.readtext(test_image, batch_size=args.recognizer_batch_size))

print('Profiling torch...')
fps_torch = profile_reader(reader)

reader.detector.module = detector_trt
reader.recognizer.module = recognizer_trt


print('Dumping TensorRT output...')
print(reader.readtext(test_image, batch_size=args.recognizer_batch_size))

print('Profiling torch...')
fps_trt = profile_reader(reader)


print(f'FPS Torch: {fps_torch}')
print(f'FPS TensorRT: {fps_trt}')