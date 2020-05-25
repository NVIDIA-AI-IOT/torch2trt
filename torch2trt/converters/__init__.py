# dummy converters throw warnings method encountered
import tensorrt as trt
from .dummy_converters import *
from torch2trt.utils import get_trt_version

# supported converters will override dummy converters
trt_version = get_trt_version()

from .activation import *
from .adaptive_avg_pool2d import *
from .adaptive_max_pool2d import *
from .AdaptiveAvgPool2d import *
from .add import *
from .mul import *
from .div import *
from .BatchNorm1d import *
from .cat import *
from .clamp import *
from .Conv1d import *
from .getitem import *
from .identity import *
from .Identity import *
from .instance_norm import *
from .Linear import *
from .LogSoftmax import *
from .max_pool2d import *
from .max import *
from .min import *
from .normalize import *
from .pad import *
from .permute import *
from .pow import *
from .prelu import *
from .prod import *
from .relu import *
from .ReLU import *
from .relu6 import *
from .ReLU6 import *
from .sigmoid import *
from .sub import *
from .sum import *
from .view import *
from .tanh import *
from .mean import *
from .softmax import *
from .split import *
from .chunk import *
from .unary import *

## Some ops implementation has been changed based on trt version.

if trt_version < 7.0:  ##TRT ops supported in trt 5 and 6
    from .avg_pool2d import *
    from .BatchNorm2d import *
    from .Conv2d import *
    from .ConvTranspose2d import *
    from .transpose import *

if trt_version >= 7.0:
    from .trt7_ops.avg_pool import *
    from .trt7_ops.compare import *
    from .trt7_ops.batch_norm import *
    from .trt7_ops.Conv import *
    from .trt7_ops.ConvTranspose import *
    from .trt7_ops.stack import *
    from .trt7_ops.transpose import *

## Upsample op will be fixed in 7.1 , hence a special case
if trt_version >= 7.1:
    from .upsample import *
else:
    try:
        from .interpolate import *
    except:
        pass
