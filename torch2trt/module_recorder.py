import time
import torch
import numpy as np
import contextlib
from typing import Any, List
from dataclasses import dataclass
from .flattener import Flattener


@dataclass
class ModuleRecorderInfo:
    input_shapes_flat: Any
    execution_time: float


class SingleModuleRecorder:

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.results: List[ModuleRecorderInfo] = []
        self._t0 = None
        self._pre_handle = None
        self._post_handle = None
        self._flattener = None
        
    def _pre_hook(self, module, input):
        torch.cuda.current_stream().synchronize()
        self._t0 = time.perf_counter_ns()

    def _post_hook(self, module, input, output):
        torch.cuda.current_stream().synchronize()
        execution_time = 1000 * (time.perf_counter_ns() - self._t0) / 1e9

        if self._flattener is None:
            self._flattener = Flattener.from_value(input)

        input_flat = self._flattener.flatten(input)
        input_shapes_flat = [x.shape for x in input_flat]

        result = ModuleRecorderInfo(
            input_shapes_flat=input_shapes_flat,
            execution_time=execution_time
        )

        self.results.append(result)
    
    def clear(self):
        self.results = []

    def __enter__(self, *args, **kwargs):
        self._pre_handle = self.module.register_forward_pre_hook(self._pre_hook)
        self._post_handle = self.module.register_forward_hook(self._post_hook)
        return self

    def __exit__(self, *args, **kwargs):
        self._pre_handle.remove()
        self._post_handle.remove()
        self._pre_handle = None
        self._post_handle = None

    def get_mean_execution_time(self):
        if len(self.results) == 0:
            return -1
        return np.mean([r.execution_time for r in self.results])

    def get_median_execution_time(self):
        if len(self.results) == 0:
            return -1
        return np.median([r.execution_time for r in self.results])

    def get_total_execution_time(self):
        if len(self.results) == 0:
            return -1
        return np.sum([r.execution_time for r in self.results])

    def get_max_execution_time(self):
        if len(self.results) == 0:
            return -1
        return np.max([r.execution_time for r in self.results])

    def get_min_execution_time(self):
        if len(self.results) == 0:
            return -1
        return np.min([r.execution_time for r in self.results])
        
    def get_all_input_shapes_flat(self):
        shapes = []
        for result in self.results:
            shapes.append(result.input_shapes_flat)
        return shapes

    def get_input_shapes_statistic(self, statistic, return_flat: bool = False):
        shapes = self.get_all_input_shapes_flat()
        if len(shapes) == 0:
            return []

        shape_stats_flat = []

        for tensor_idx in range(len(shapes[0])):
            tensor_shapes = [list(shapes[i][tensor_idx]) for i in range(len(shapes))]
            tensor_shapes = np.array(tensor_shapes)
            stat = statistic(tensor_shapes)
            stat = np.round(stat).astype(np.int64).tolist()
            shape_stats_flat.append(stat)

        if return_flat:
            return shape_stats_flat
        else:
            shape_stats = self._flattener.unflatten(shape_stats_flat)
            return shape_stats

    def get_min_input_shapes(self, return_flat: bool = False):
        return self.get_input_shapes_statistic(lambda x: np.amin(x, axis=0), return_flat=return_flat)

    def get_max_input_shapes(self, return_flat: bool = False):
        return self.get_input_shapes_statistic(lambda x: np.amax(x, axis=0), return_flat=return_flat)

    def get_median_input_shapes(self, return_flat: bool = False):
        return self.get_input_shapes_statistic(lambda x: np.median(x, axis=0), return_flat=return_flat)

    def get_mean_input_shapes(self, return_flat: bool = False):
        return self.get_input_shapes_statistic(lambda x: np.mean(x, axis=0), return_flat=return_flat)
            
    def get_num_calls(self):
        return len(self.results)


class ModuleRecorder:

    def __init__(self, module):
        self.module = module
        self.recorders = {}
        for name, child in self.module.named_modules():
            recorder = SingleModuleRecorder(child)
            self.recorders[name] = recorder

    def clear(self):
        for _, r in self.recorders.items():
            r.clear()

    def __enter__(self, *args, **kwargs):
        for name, recorder in self.recorders.items():
            recorder.__enter__(*args, **kwargs)
        return self

    def __exit__(self, *args, **kwargs):
        for name, recorder in self.recorders.items():
            recorder.__exit__(*args, **kwargs)
        return self

    def stats(self):
        stats = {}
        for name, rec in self.recorders.items():
            stat_for_rec = {}
            stat_for_rec['mean_execution_time'] = rec.get_mean_execution_time()
            stat_for_rec['median_execution_time'] = rec.get_median_execution_time()
            stat_for_rec['min_execution_time'] = rec.get_min_execution_time()
            stat_for_rec['max_execution_time'] = rec.get_max_execution_time()
            stat_for_rec['total_execution_time'] = rec.get_total_execution_time()
            stat_for_rec['min_input_shapes'] = rec.get_min_input_shapes()
            stat_for_rec['max_input_shapes'] = rec.get_max_input_shapes()
            stat_for_rec['mean_input_shapes'] = rec.get_mean_input_shapes()
            stat_for_rec['median_input_shapes'] = rec.get_median_input_shapes()
            stat_for_rec['num_calls'] = rec.get_num_calls()
            stats[name] = stat_for_rec
        return stats