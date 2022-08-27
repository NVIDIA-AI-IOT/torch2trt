# torch2trt EasyOCR Example

This example uses torch2trt to optimize EasyOCR.  EasyOCR is split into
two TensorRT engines, one for the detector, one for the recognizer.

To run the example, follow these steps

1. Download example images

    ```bash
    ./download_images.sh
    ```

2. Generate data for shape inference and calibration.  By default this script will look in the ``images`` directory.

    ```bash
    python3 generate_data.py
    ```

3. Optimize the Text Detector.  This will use the data from step 1 for shape inference and calibration.  It creates a file ``detector_trt.pth``.

    ```bash
    python3 optimizer_detector.py
    ```

4. Optimize the Text Recognizer. This also uses the data generated from step 1.  It creates a file ``recognizer_trt.pth``.

    ```bash
    python3 optimize_recognizer.py

5. Run the pipeline end to end and compare the performance to the original PyTorch model. 

    ```bash
    python3 run_end2end.py
    ```

That's it!  To use the model in your application, reference these scripts for more details.  Specifically, reference
``run_end2end.py`` to see how to create and execute the full model pipeline.