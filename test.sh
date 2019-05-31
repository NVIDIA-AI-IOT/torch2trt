OUTPUT_FILE=$1

touch $OUTPUT_FILE

echo "| Name | Data Type | Input Shapes | torch2trt kwargs | Max Error | FPS (PyTorch) | FPS (TensorRT) |" >> $OUTPUT_FILE
echo "|------|-----------|--------------|------------------|-----------|---------------|----------------|" >> $OUTPUT_FILE

#python3 -m torch2trt.test -o $OUTPUT_FILE --name alexnet
#python3 -m torch2trt.test -o $OUTPUT_FILE --name squeezenet1_0
#python3 -m torch2trt.test -o $OUTPUT_FILE --name squeezenet1_1
#python3 -m torch2trt.test -o $OUTPUT_FILE --name resnet18
#python3 -m torch2trt.test -o $OUTPUT_FILE --name resnet34
#python3 -m torch2trt.test -o $OUTPUT_FILE --name resnet50
#python3 -m torch2trt.test -o $OUTPUT_FILE --name resnet101
#python3 -m torch2trt.test -o $OUTPUT_FILE --name resnet152
#python3 -m torch2trt.test -o $OUTPUT_FILE --name densenet121
#python3 -m torch2trt.test -o $OUTPUT_FILE --name densenet169
#python3 -m torch2trt.test -o $OUTPUT_FILE --name densenet201
#python3 -m torch2trt.test -o $OUTPUT_FILE --name densenet161
python3 -m torch2trt.test -o $OUTPUT_FILE --name vgg11$
python3 -m torch2trt.test -o $OUTPUT_FILE --name vgg13$
python3 -m torch2trt.test -o $OUTPUT_FILE --name vgg16$
python3 -m torch2trt.test -o $OUTPUT_FILE --name vgg19$
#python3 -m torch2trt.test -o $OUTPUT_FILE --name vgg11_bn
#python3 -m torch2trt.test -o $OUTPUT_FILE --name vgg13_bn
#python3 -m torch2trt.test -o $OUTPUT_FILE --name vgg16_bn
#python3 -m torch2trt.test -o $OUTPUT_FILE --name vgg19_bn
