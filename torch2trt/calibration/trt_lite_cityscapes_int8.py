import glob, os
from random import shuffle
import numpy as np
from PIL import Image
import pycuda.driver as cuda

import tensorrt as trt

import labels
import calibrator

MEAN = (71.60167789, 82.09696889, 72.30508881)
MODEL_DIR = 'data/fcn8s/'
CITYSCAPES_DIR = '/data/shared/Cityscapes/'
TEST_IMAGE = CITYSCAPES_DIR + 'leftImg8bit/val/lindau/lindau_000042_000019_leftImg8bit.png'
CALIBRATION_DATASET_LOC = CITYSCAPES_DIR + 'leftImg8bit/train/*/*.png'

CLASSES = 19
CHANNEL = 3
HEIGHT = 500
WIDTH = 500

logger = trt.Logger(trt.Logger.ERROR)

def sub_mean_chw(data):
    data = data.transpose((1, 2, 0)) # CHW -> HWC
    data -= np.array(MEAN) # Broadcast subtract
    data = data.transpose((2, 0 ,1)) # HWC -> CHW
    return data

def color_map(output):
    output = output.reshape(CLASSES, HEIGHT, WIDTH)
    out_col = np.zeros(shape = (HEIGHT, WIDTH), dtype = (np.uint8, 3))
    for x in range(WIDTH):
        for y in range(HEIGHT):
            out_col[y, x] = labels.id2label[labels.trainId2label[np.argmax(output[:, y, x])].id].color
    return out_col

def create_calibration_dataset():
    # Create list of calibration images
    # This sample code picks 100 images at random from training set
    calibration_files = glob.glob(CALIBRATION_DATASET_LOC)
    shuffle(calibration_files)
    return calibration_files[:100]

def get_engine(int8_calibrator, engine_file_path=""):
    if os.path.exists(engine_file_path):
        with open(engine_file_path, 'rb') as f, trt.Runtime(logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print("building engine...")
        with trt.Builder(logger) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
            builder.max_batch_size=1
            builder.max_workspace_size=(256 << 20)
            builder.int8_mode=True
            builder.int8_calibrator=int8_calibrator
            builder.strict_type_constraints = True

            if not os.path.exists(MODEL_DIR + 'fcn8s.prototxt'):
                print("There is no prototxt at: %s"%(MODEL_DIR + 'fcn8s.prototxt'))
                exit(0)
            parser.parse(deploy=MODEL_DIR + 'fcn8s.prototxt', model=MODEL_DIR + 'fcn8s.caffemodel', network = network, dtype=trt.float32)
            network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
            engine = builder.build_cuda_engine(network)
        return engine

def get_engine2(engine_file_path=""):
    if os.path.exists(engine_file_path):
        with open(engine_file_path, 'rb') as f, trt.Runtime(logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print("building engine...")
        with trt.Builder(logger) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
            builder.max_batch_size=1
            builder.max_workspace_size=(256 << 20)
            builder.fp16_mode=False
            builder.strict_type_constraints = True

            if not os.path.exists(MODEL_DIR + 'fcn8s.prototxt'):
                print("There is no prototxt at: %s"%(MODEL_DIR + 'fcn8s.prototxt'))
                exit(0)
            parser.parse(deploy=MODEL_DIR + 'fcn8s.prototxt', model=MODEL_DIR + 'fcn8s.caffemodel', network = network, dtype=trt.float32)
            network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
            engine = builder.build_cuda_engine(network)
        return engine

def do_inference(test_data, engine, stream):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    h_input = h_input.reshape(3, 500, 500)
    h_output = h_output.reshape(19, 500, 500)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    np.copyto(h_input, test_data)

    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context = engine.create_execution_context()
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()

    out = color_map(h_output)
    return out

def main():
    calibration_files = create_calibration_dataset()

    # Process 5 images at a time for calibration
    # This batch size can be different from MaxBatchSize (1 in this example)
    print("Ready ImageBatchStream...")
    batchstream = calibrator.ImageBatchStream(5, calibration_files, sub_mean_chw)
    print("Stream ready done!")
    print("Ready Entropy Calibration...")
    int8_calibrator = calibrator.pyEntropyCalibrator(["data"], batchstream, 'data/calibration_cache.bin')
    print("Calibrator ready done!")

    # Build engine
    engine1 = get_engine(int8_calibrator)
    engine2 = get_engine2()

    # Predict
    test_data = calibrator.ImageBatchStream.read_image(TEST_IMAGE)
    test_data = sub_mean_chw(test_data)
    stream = cuda.Stream()

    out1 = do_inference(test_data, engine1, stream)
    out2 = do_inference(test_data, engine2, stream)

    test_img = Image.fromarray(out1, 'RGB')
    test_img.save("Int8_inference", "jpeg")
    test_img = Image.fromarray(out2, 'RGB')
    test_img.save("Float_inference", "jpeg")

if __name__ == "__main__":
    main()
