import torch
import tensorrt as trt


class TensorBatchDataset():
    
    def __init__(self, tensors):
        self.tensors = tensors
    
    def __len__(self):
        return len(self.tensors[0])
    
    def __getitem__(self, idx):
        return [t[idx] for t in self.tensors]
    
    
class DatasetCalibrator(trt.IInt8Calibrator):
    
    def __init__(self, dataset, batch_size=1, algorithm=trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2):
        super().__init__()
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.algorithm = algorithm
        
        # pull sample, should not include batch dimension
        inputs = dataset[0] 
        
        # create buffers that will hold random data batches
        self.buffers = []
        for tensor in inputs:
            size = (batch_size,) + tuple(tensor.shape)
            buf = torch.randn(size=size, dtype=tensor.dtype, device=tensor.device).contiguous()
            self.buffers.append(buf)
            
        self.count = 0
        
    def get_batch(self, *args, **kwargs):
        if self.count < len(self.dataset):
            
            for i in range(self.batch_size):
                
                idx = self.count % len(self.dataset) # roll around if not multiple of dataset
                inputs = self.dataset[idx]
                
                for j, tensor in enumerate(inputs):
                    self.buffers[j][i].copy_(tensor)
                
                self.count += 1
                
            return [int(buf.data_ptr()) for buf in self.buffers]
        else:
            return []
        
    def get_algorithm(self):
        return self.algorithm
    
    def get_batch_size(self):
        return self.batch_size
    
    def read_calibration_cache(self):
        return None
    
    def write_calibration_cache(self, cache):
        pass