import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import ctypes
import tensorrt as trt
import os
import torch.utils.data as data

CHANNEL = 3
HEIGHT = 500
WIDTH = 500

class pyEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, input_layers, stream, cache_path):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.input_layers = input_layers
        self.stream = stream
        self.cache_path = cache_path
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, bindings, names):
        batch = self.stream.next_batch()
        if not batch.size:
            return None

        cuda.memcpy_htod(self.d_input, batch)
        for i in self.input_layers[0]:
            assert names[0] != i

        #bindings[0] = int(self.d_input)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_path, 'wb') as f:
            f.write(cache)
        return None

# example data loader for citiscapes
class ImageBatchStream(data.Dataset):
    def __init__(self, batch_size, calibration_files, preprocessor):
        self.batch_size = batch_size
        self.max_batches = (len(calibration_files) // batch_size) + \
                            (1 if (len(calibration_files) % batch_size) else 0)
        self.files = calibration_files

        CHANNEL = 3
        HEIGHT = 500
        WIDTH = 500

        self.calibration_data = np.zeros((batch_size, CHANNEL, HEIGHT, WIDTH), dtype = np.float32)
        self.batch = 0
        self.preprocessor = preprocessor

    @staticmethod
    def read_image(path):
        img = Image.open(path).resize((500, 500), Image.NEAREST)
        im = np.array(img, dtype=np.float32, order='C')
        im = im[:, :, ::-1]
        im = im.transpose((2, 0, 1))
        return im

    def reset(self):
        self.batch = 0

    def next_batch(self):
        if self.batch < self.max_batches:
            imgs = []
            files_for_batch = self.files[self.batch_size * self.batch : self.batch_size * (self.batch + 1)]
            for f in files_for_batch:
                print("[ImageBatchStream %dth] Processing "%self.batch, f)
                img = ImageBatchStream.read_image(f)
                img = self.preprocessor(img)
                imgs.append(img)

            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1
            return np.ascontiguousarray(self.calibration_data, dtype = np.float32)
        else:
            return np.array([])


