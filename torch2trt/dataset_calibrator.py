import torch
import tensorrt as trt
import os
from .flattener import Flattener

__all__ = [
    'DEFAULT_CALIBRATION_ALGORITHM',
    'DatasetCalibrator'
]


if trt.__version__ >= '5.1':
    DEFAULT_CALIBRATION_ALGORITHM = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
else:
    DEFAULT_CALIBRATION_ALGORITHM = trt.CalibrationAlgoType.ENTROPY_CALIBRATION


class DatasetCalibrator(trt.IInt8Calibrator):

    def __init__(self, dataset, algorithm=DEFAULT_CALIBRATION_ALGORITHM, cache_file=None, flattener=None):
        super(DatasetCalibrator, self).__init__()
        self.dataset = dataset
        self.algorithm = algorithm
        self.count = 0
        self.cache_file = cache_file
        if flattener is None:
            flattener = Flattener.from_value(dataset[0])
        self.flattener = flattener

    def get_batch(self, *args, **kwargs):
        if self.count < len(self.dataset):
            tensors = self.flattener.flatten(self.dataset[self.count])
            bindings = [int(t.data_ptr()) for t in tensors]
            self.count += 1
            return bindings
        else:
            return []

    def get_algorithm(self):
        return self.algorithm

    def get_batch_size(self):
        return 1

    def read_calibration_cache(self, *args, **kwargs):
        if (self.cache_file is not None) and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
                
    def write_calibration_cache(self, cache, *args, **kwargs):
        if self.cache_file is not None:
            with open(self.cache_file, 'wb') as f:
                f.write(cache)
