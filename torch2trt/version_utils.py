import packaging.version
import tensorrt as trt
import torch


def trt_version():
    return Version(trt.__version__)


def torch_version():
    return Version(torch.__version__)


class Version(packaging.version.Version):

    def __ge__(self, other):
        if isinstance(other, str):
            other = Version(other)
        return super().__ge__(other)

    def __le__(self, other):
        if isinstance(other, str):
            other = Version(other)
        return super().__le__(other)

    def __eq__(self, other):
        if isinstance(other, str):
            other = Version(other)
        return super().__eq__(other)

    def __gt__(self, other):
        if isinstance(other, str):
            other = Version(other)
        return super().__gt__(other)
    
    def __lt__(self, other):
        if isinstance(other, str):
            other = Version(other)
        return super().__lt__(other)
    