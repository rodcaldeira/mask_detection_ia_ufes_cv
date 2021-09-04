# Mask Detection Project for Visual Cognition Class
#### Ufes - 2021/1
###### Cezar Gobbo e Rodrigo Caldeira
#
###### Dependencies
- [CUDA](https://developer.nvidia.com/cuda-zone) - Parallel Computing Platform and API model v10.1
- [YoloV4](https://github.com/AlexeyAB/darknet) - Convolutional Neural Network (CNN)
- [OpenCV3](https://github.com/opencv/opencv.git) - Computer Vision library v3.4.0
- [CMake](https://cmake.org) - Compilation Process control software

#### Ubuntu 18.04 LTS (64 bit) Instalation Guide

##### Yolov4 Instalação and Configuration
```sh
git clone https://github.com/AlexeyAB/darknet
cd darknet
git checkout darknet_yolo_v4_pre
```
Edit `Makefile` replacing lines below ("-" means remove, "+" means add)
```sh
-GPU=0
+GPU=1
-OPENCV=0
+OPENCV=1
-LIBSO=0
+LIBSO=1
-NVCC=nvcc
+NVCC=/usr/local/cuda-10.1/bin/nvcc
-LDFLAGS+= `pkg-config --libs opencv4 2> /dev/null || pkg-config --libs opencv`
-COMMON+= `pkg-config --cflags opencv4 2> /dev/null || pkg-config --cflags opencv`
+LDFLAGS+= `pkg-config --libs opencv3 2> /dev/null || pkg-config --libs opencv`
+COMMON+= `pkg-config --cflags opencv3 2> /dev/null || pkg-config --cflags opencv`
-COMMON+= -DGPU -I/usr/local/cuda/include/
+COMMON+= -DGPU -I/usr/local/cuda-10.1/include/
```

After edit and save, you must build the darknet YoloV4 with the command:
```sh
make
```
After gen the file `libdarknet.so` we need to move it to our folder `lib/`
#
##### Building Project

Download models at LINK_DOWNLOAD
Save them on `model/` folder at repository root level.

```sh
mkdir build && cd build
cmake ..
make
```
#
##### Usage
USAGE_DETAILS
