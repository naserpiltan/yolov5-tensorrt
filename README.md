### YOLOv5-TensorRT

![](./docs/demo.png)
  
The goal of this library is to provide an accessible and robust method for performing efficient, real-time object detection with [YOLOv5](https://github.com/ultralytics/yolov5) using NVIDIA TensorRT. The library was developed with real-world deployment and robustness in mind. Moreover, the library is extensively documented and comes with various guided examples.



## <div align="center">Features</div>

- FP32 and FP16 inference
- Batch inference
- Support for varying input dimensions
- ONNX support
- CUDA-accelerated pre-processing
- Integration with OpenCV (with optionally also the OpenCV-CUDA module)
- Extensive documentation available on all classes, methods and functions


## <div align="center">Usage</div>

<details>
<summary>Command-line Usage</summary>
In addition to the C++ API, the library also comes with various tools/demos. Assuming that your YOLOv5 model is stored as <em>yolov5.onnx</em>, you can build a TensorRT engine using:
  
```bash
./build_engine --input yolov5.onnx --output yolov5.engine
```
The resulting engine will be stored to disk at <em>yolov5.engine</em>. For an overview of all available options for this tool, see [build_engine](examples/builder).
  
After the engine has been stored, you can load it and detect objects as following:
```bash
./process_image --engine yolov5.engine --input image.png --output result.png
```
A visualization of the result will be stored to disk at <em>result.png</em>. For an overview of all available options for this tool, see [process_image](examples/image).
  
</details>

<details open>
<summary>C++ usage</summary>
This example assumes that your YOLOv5 model is stored as <em>yolov5.onnx</em>. Include <em>yolov5_builder.hpp</em> in your project, and build the TensorRT engine in just three lines of C++ code:
  
```cpp
yolov5::Builder builder;
builder.init();
builder.build("yolov5.onnx", "yolov5.engine");
```

  Next, detect objects with YOLOv5 using the following code:
```cpp
yolov5::Detector detector;
detector.init();
detector.loadEngine("yolov5.engine");

cv::Mat image = cv::imread("image.png");

std::vector<yolov5::Detection> detections;
detector.detect(image, &detections);
```
  
</details>

<details>
<summary>Examples</summary>

Various **documented** examples can be found in the [examples](examples) directory.

In order to **build** a TensorRT engine based on an ONNX model, the following
tool/example is available:
- [build_engine](examples/builder): build a TensorRT engine based on your ONNX model

For **object detection**, the following tools/examples are available:
- [process_image](examples/image): detect objects in a single image
- [process_live](examples/live): detect objects live in a video stream (e.g. webcam)
- [process_batch](examples/batch): detect objects in multiple images (batch inference)
  
</details>


## <div align="center">Install</div>

<details>
<summary>Platforms</summary>
  
The library can be used on:
- Most modern linux distributions
- NVIDIA L4T (Jetson platform)

Moreover, only 2 dependencies are needed:
- TensorRT >=8 (libnvinfer libnvonnxparsers-dev)
- OpenCV
  
</details>
  
<details>
<summary>Building from source</summary>
  
The software can be compiled using CMake and a modern C++ compiler (e.g. GCC)
with support for C++14, using the following steps:

```bash
mkdir build
cd build
cmake ..
make
```
</details>
  

## <div align="center">About</div>

This library was originally developed for [VDL RobotSports](https://robotsports.nl),
an industrial team based in the Netherlands participating in the RoboCup Middle
Size League, and currently sees active use on the soccer robots.

If you like this library and would like to cite it, please use the following (LateX):

```tex
@misc{yolov5tensorrt,
  author       = {van der Meer, Noah and van Hoof, Charel},
  title        = {{yolov5-tensorrt}: Real-time object detection with {YOLOv5} and {TensorRT}},
  howpublished = {GitHub},
  year         = {2021},
  note         = {\url{https://github.com/noahmr/yolov5-tensorrt}}
}
```

## <div align="center">License</div>

Copyright (c) 2021, Noah van der Meer

This software is licenced under the MIT license, see [LICENCE.md](LICENCE.md).
