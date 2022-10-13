import os
import timm
import torch
import time
import json
from torch2trt import torch2trt, TRTModule, trt
from dataclasses import dataclass, asdict
from argparse_dataclass import ArgumentParser
from typing import Literal
from enum import Enum


class Status:
    STARTING = "STARTING"
    PROFILING = "PROFILING"
    FINISHED = "FINISED"


def profile_qps(model, data, num_warmup, num_profile):
    for _ in range(num_warmup):
        out = model(data)

    torch.cuda.current_stream().synchronize()
    t0 = time.perf_counter()
    for _ in range(num_profile):
        out = model(data)
    torch.cuda.current_stream().synchronize()
    t1 = time.perf_counter()

    return num_profile / (t1 - t0)


def profile_latency(model, data, num_warmup, num_profile):
    for _ in range(num_warmup):
        out = model(data)

    torch.cuda.current_stream().synchronize()
    t0 = time.perf_counter()
    for _ in range(num_profile):
        out = model(data)
        torch.cuda.current_stream().synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / num_profile

def log_level_to_trt(log_level: str):
    if log_level == "verbose":
        return trt.Logger.VERBOSE
    elif log_level  == "info":
        return trt.Logger.INFO
    elif log_level == "error":
        return trt.Logger.ERROR
    elif log_level == "warning":
        return trt.Logger.WARNING
    elif log_level == "internal_error":
        return trt.Logger.INTERNAL_ERROR
    else:
        raise ValueError(f"Unknown log level: {log_level}")

@dataclass
class Args:
    model: str
    output_dir: str = "data/timm"
    batch_size: int = 1
    fp16_mode: bool = False
    int8_mode: bool = False
    size: int = 224
    pretrained: bool = True
    save_engine: bool = True
    num_warmup: int = 10
    num_profile: int = 100
    use_cached_engine: bool = True
    log_level: Literal["verbose", "info", "error", "warning", "internal_error"] = "error"

    def id(self):
        fn = self.model
        if self.pretrained:
            fn += "-pre"
        fn += "-trt"
        fn += f"-{self.batch_size}"
        fn += f"-{self.size}"
        if self.fp16_mode:
            fn += "-fp16"
        if self.int8_mode:
            fn += "-int8"
        return fn

    def engine_filepath(self):
        return os.path.join(self.output_dir, self.id() + ".pth")

    def metadata_filepath(self):
        return os.path.join(self.output_dir, self.id() + ".json")

    def write_output(self, output):
        with open(self.metadata_filepath(), 'w') as f:
            json.dump(output, f, indent=2)

    def run(self):
        with torch.no_grad():
            model = timm.create_model(self.model, pretrained=self.pretrained)
            model = model.cuda().eval()
            data = torch.randn(self.batch_size, 3, self.size, self.size).cuda()

            output = {}
            output['args'] = asdict(self)
            output['status'] = str(Status.STARTING)
            self.write_output(output)

            if self.use_cached_engine and os.path.exists(self.engine_filepath()):
                print("Loading cached engine...")
                model_trt = TRTModule()
                model_trt.load_state_dict(torch.load(self.engine_filepath()))
            else:
                print("Building engine...")
                model_trt = torch2trt(
                    model, 
                    [data], 
                    fp16_mode=self.fp16_mode, 
                    int8_mode=self.int8_mode,
                    log_level=log_level_to_trt(self.log_level)
                )
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)

            if self.save_engine:
                print(f"Saving engine to {self.engine_filepath()}...")
                torch.save(model_trt.state_dict(), self.engine_filepath())

            output['status'] = str(Status.PROFILING)
            self.write_output(output)
            data = torch.randn(self.batch_size, 3, self.size, self.size).cuda()

            print(f"Profiling PyTorch...")
            qps_torch = profile_qps(model, data, self.num_warmup, self.num_profile)
            latency_torch = profile_latency(model, data, self.num_warmup, self.num_profile)
            fps_torch = qps_torch * self.batch_size

            print(f"Profiling TensorRT...")
            qps_trt = profile_qps(model_trt, data, self.num_warmup, self.num_profile)
            latency_trt = profile_latency(model_trt, data, self.num_warmup, self.num_profile)
            fps_trt = qps_trt * self.batch_size
            
            data = torch.randn(self.batch_size, 3, self.size, self.size).cuda()
            dout = model(data)
            dout_trt = model_trt(data)
            max_abs_error = float(torch.max(torch.abs(dout - dout_trt)))

            result = {}
            result['latency_torch'] = latency_torch
            result['latency_trt'] = latency_trt
            result['fps_torch'] = fps_torch
            result['fps_trt'] = fps_trt
            result['max_abs_error'] = max_abs_error
            output['results'] = result
            output['status'] = str(Status.FINISHED)
            self.write_output(output)

if __name__ == "__main__":
    parser = ArgumentParser(Args)
    args = parser.parse_args()
    print(json.dumps(asdict(args), indent=2))
    args.run()