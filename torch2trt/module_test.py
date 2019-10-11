import torch
import torchvision
import torch.utils.data as data
import glob, os, sys
from random import shuffle
from .calibration import Transformer
from PIL import Image
import numpy as np

preprocess = Transformer(size = 224, mean = [0, 0, 0])

class dataLoader(data.Dataset):
    def __init__(self, data_path, batch_size=1, preprocess=None, n=100):
        """
            batch_size: here, batch_size is different from the batch for inference.
                This is for calibration.
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.preprocess = preprocess
        if preprocess is not None:
            self.size = preprocess.size
        else:
            self.size = (224, 224)

        self.files = glob.glob(os.path.join(self.data_path, '*'))
        shuffle(self.files)
        self.files = self.files[:n]
        self.max_batches = (len(self.files) // self.batch_size) + \
                            (1 if (len(self.files) % self.batch_size) else 0)

        self.calibration_data = np.zeros((batch_size, 3, self.size[0], self.size[1]), dtype=np.float32)
        self.batch = 0

    def read_image(self, path):
        return np.asarray(Image.open(path))

    def reset(self):
        self.batch = 0

    def next_batch(self):
        if self.batch < self.max_batches:
            imgs = []
            files_for_batch = self.files[self.batch_size * self.batch : self.batch_size * (self.batch + 1)]
            sys.stdout.write("\r [*] calibration is proceeding(%.2f%%)"%((self.batch+1)/self.max_batches*100))
            sys.stdout.flush()
            for f in files_for_batch:
                img = self.read_image(f)
                img = self.preprocess(img)
                imgs.append(img.transpose(2, 0 ,1))

            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

class ModuleTest(object):
    def __init__(self, module_fn, dtype, device, input_shapes, **torch2trt_kwargs):
        self.module_fn = module_fn
        self.dtype = dtype
        self.device = device
        self.input_shapes = input_shapes
        self.torch2trt_kwargs = torch2trt_kwargs
        
    def module_name(self):
        return self.module_fn.__module__ + '.' + self.module_fn.__name__


MODULE_TESTS = [
    ModuleTest(torchvision.models.alexnet, torch.float16, torch.device('cuda'), [(1, 3, 224, 224)], int8_mode=True, int8_stream = dataLoader('/data/shared/Cityscapes/leftImg8bit/train/*/', 5, preprocess))
]


def add_module_test(dtype, device, input_shapes, **torch2trt_kwargs):
    def register_module_test(module):
        global MODULE_TESTS
        MODULE_TESTS += [ModuleTest(module, dtype, device, input_shapes, **torch2trt_kwargs)]
        return module
    return register_module_test
