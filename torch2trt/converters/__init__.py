# dummy converters throw warnings method encountered
import tensorrt as trt 
from .dummy_converters import *

# supported converters will override dummy converters

from .activation import *
from .adaptive_avg_pool2d import *
from .adaptive_max_pool2d import *
from .AdaptiveAvgPool2d import *
from .add import *
from .avg_pool import *
from .mul import *
from .compare import *
from .div import *
from .BatchNorm1d import *
from .batch_norm import *
from .cat import *
from .clamp import *
from .Conv1d import *
from .Conv import *
from .ConvTranspose import *
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
from .stack import *
from .sub import *
from .sum import *
from .view import *
from .tanh import *
from .transpose import * 
from .mean import *
from .softmax import *
from .split import *
from .chunk import *
from .unary import *

if int(trt.__version__.split(".")[0]) >= 6:
    print("using tensorrt version {}".format(trt.__version__.split(".")[0]))
    from .upsample import *
else:
    print("using tensorrt version {}".format(trt.__version__.split(".")[0]))
    try:
        from .interpolate import *
    except:
        pass

