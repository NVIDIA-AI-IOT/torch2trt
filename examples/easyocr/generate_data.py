from argparse import ArgumentParser
import cv2
import torch
import glob
from easyocr import Reader
from torch2trt.dataset import FolderDataset
from torch2trt import torch2trt, TRTModule
import math
import os

parser = ArgumentParser()
parser.add_argument('--images', type=str, default='images')
parser.add_argument('--detector_data', type=str, default='detector_data')
parser.add_argument('--recognizer_data', type=str, default='recognizer_data')
parser.add_argument('--max_image_area', type=int, default=1280*720)
parser.add_argument('--recognizer_batch_size', type=int, default=1)
args = parser.parse_args()


reader = Reader(['en'])


detector_dataset = FolderDataset(args.detector_data)
recognizer_dataset = FolderDataset(args.recognizer_data)


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


with detector_dataset.record(reader.detector.module):
    with recognizer_dataset.record(reader.recognizer.module):
        
        for path in glob.glob(os.path.join(args.images, '*.jpg')):
            print(path)

            image = cv2.imread(path)

            image = shrink_to_area(image, args.max_image_area)

            reader.readtext(image, batch_size=args.recognizer_batch_size)
