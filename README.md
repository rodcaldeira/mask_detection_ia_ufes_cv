# Mask Detection Project for Visual Cognition Class
#### Ufes - 2021/1
###### Cezar Augusto Gobbo Passamani e Rodrigo Caldeira
#

#### Ubuntu 18.04 LTS (64 bit) Instalation Guide

##### Dependencies

Cmake
```
mkdir ~/libraries
cd ~/libraries
git clone https://gitlab.kitware.com/cmake/cmake.git
cd cmake
./bootstrap
make
sudo make install
sudo ldconfig
cd ~/libraries
```

Nvidia Driver 440 Installation (Only For Nvidia PCs)
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update

Now go to Software & Updates -> Additional Drivers -> Choose Latest Driver (careful, do not choose the Server one)
Wait installation
Reboot
```

Nvidia Cuda Installation (Only For Nvidia PCs)
```
sudo apt-get install freeglut3-dev 
sudo apt-get install libx11-dev 
sudo apt-get install libxmu-dev 
sudo apt-get install libxi-dev 
sudo apt-get install libglu1-mesa 
sudo apt-get install libglu1-mesa-dev

mkdir ~/libraries/
cd ~/libraries/
wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.168_418.67_linux.run
sudo ./cuda_10.1.168_418.67_linux.run (ps: unmark the nvidia driver option when asked)

echo "#AS1 VARS
export PATH=$PATH:/usr/local/cuda-10.1/bin\"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64\"" >> ~/.bashrc

# Download Cudnn
wget https://developer.download.nvidia.com/compute/redist/cudnn/v7.6.1/cudnn-10.1-linux-x64-v7.6.1.34.tgz
tar -xzvf cudnn-10.1-linux-x64-v7.6.1.34.tgz
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-10.1/include 
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-10.1/lib64/ 
sudo chmod a+r /usr/local/cuda-10.1/lib64/libcudnn*
```

OpenCV 3.4.0 Installation
```sh
mkdir ~/libraries
cd ~/libraries
mkdir opencv3.4.0 && cd opencv3.4.0
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 3.4.0
cd ../opencv
git checkout 3.4.0
mkdir build && cd build
cmake -DWITH_EIGEN=ON -DWITH_V4L=ON -DWITH_CUDA=ON -DWITH_CUBLAS=ON -DWITH_TBB=ON -DWITH_OPENGL=ON -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" -DBUILD_opencv_cudacodec=OFF -DENABLE_PRECOMPILED_HEADERS=ON -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=~/libraries/opencv3.4.0/opencv_contrib/modules  ~/libraries/opencv3.4.0/opencv/
make -j12
sudo make install
sudo ldconfig
cd ~/libraries
```

Darknet 
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

After edit and save, you must build the darknet with the command:
```sh
make
```
After compilling, a `libdarknet.so` file will be generated in the darknet root folder. 

Now go the Mask Detection project root folder and create a directory named `lib/`, then move the `libdarknet.so` into the `lib/`.
#
##### Building Project

Download the network models at https://drive.google.com/file/d/101FbAVvSUsh8zXeosQXWdMWwi7n8-2v2/view?usp=sharing.
Save them and extract at repository root level, generating a folder called `net_models` automatically.

```sh
cd mask_detection_ia_ufes_cv
mkdir build && cd build
cmake ..
make
```
#
##### Usage
```sh
Usage: ./mask_detection [--camera CAMERA] [--file FILE] [--path PATH] [--threshold THRESHOLD] [--width WIDTH] [--height HEIGHT] [--detection-off]
```
`[--camera CAMERA]` means any camera attached to the PC, just pass as an argument using `-c`. Example:
```
./mask_detection -c 0
```
`[--file FILE]` means if you have any video file to run, just pass as an argument using `-f`. Example
```
./mask_detection -f video.mp4
```
`[--path PATH]` means if you have a folder containing many videos, just pass the folder path as an argument using `-p`. Example:
```
./mask_detection -p /path/to/videos/
```
`[--threshold THRESHOLD]` is used to set the detection probability threshold of the mask recognition, if the threshold is closed to 0 means you are putting the confidence more in the YoLo network, otherwise, if the threshold is closed to 1, you are setting a lower bound to the network accuracy. To use this threshold just pass as an argument using `-t`. Example:
```
./mask_detection -p /path/to/videos/ -t 0.25
```
`[--width WIDTH] [--height HEIGHT]` means the width and the height of the cam or video you are passing as an argument. To set it, just pass the values as arguments using `-w` and `-h`. It's highly recommended to used low values depending on your machine. If you have a high end machine its okay to set FullHD or HD, otherwise you can try (640,320) Example:
```
./mask_detection -p /path/to/videos/ -w 640 -h 320
```
`[--detection-off]` is used to disable the mask detection, in case you want to. To use this argument just pass the sentence as it is. Example:
```
./mask_detection -p /path/to/videos/ --detection-off
```
