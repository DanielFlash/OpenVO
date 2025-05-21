# OpenVO: open library for Visual Odometry, Visual Navigation with Image Processing and Reinforcement Learning tools

This project provides a collection of CPP and Python modules for various tasks including image processing (histogram equalization, homography, affine transformations), object detection using ONNX and PyTorch models, coordinate calculations, map analysis, video stream processing, and base components for Reinforcement Learning (RL) agents.

## Core Functionality

*   **Video Processing (`video_processor_ovo.py`)**:
    *   Processes video streams or files frame by frame.
    *   Detects keypoints (ORB) and computes descriptors.
    *   Estimates affine transformations between consecutive frames for motion tracking.
    *   Integrates with the `Trajectory` class to update pose based on visual odometry.  
*   **Coordinate Systems and Calculations (`coord_calculator.py`, `ovo_types.py`, `trajectory_ovo.py`)**:
    *   Calculates object coordinates on a surface map based on image detections and geographical metadata.
    *   Determines overall map boundaries.
    *   Calculates local object coordinates from camera parameters, orientation, and drone position.
    *   Handles transformations between different coordinate systems (e.g., Decart to Lat/Long).
    *   `Trajectory` class for managing drone/camera pose and movement.  
* **Map Analysis and Navigation Correction (`map_analysis.py`)**:
    *   Loads and processes surface map data (raw image metadata, labeled object data).
    *   Matches locally detected objects from a drone's camera feed against a global surface map.
    *   Calculates positional deltas (corrections) using template matching on object distributions.
    *   Object verification and tracking across frames.
* **Image Processing (`combined_imgproc.py`)**:
    *   Custom CLAHE (Contrast Limited Adaptive Histogram Equalization) and global histogram equalization.
    *   Homography and Affine transformation calculations (direct linear, SVD-based).
    *   Custom RANSAC implementation for robust model fitting.
    *   Custom KNN (K-Nearest Neighbors) matching for descriptors.
    *   Data structures for points, matches, and descriptors.
*   **Object Detection (`detector.py`, `inferencers/`)**:
    *   Wrapper for object detection models (ONNX and PyTorch).
    *   Supports CUDA for accelerated inference.
    *   Includes base inferencer class and specific implementations for ONNX (`onnx_inferencer.py`) and PyTorch (`torch_inferencer.py`).
*   **Reinforcement Learning (`RL_module.py`)**:
    *   Base classes for various RL components:
        *   Models (DQN, Policy Gradient, SARSA, A2C Actor/Value, PPO Actor/Value) using PyTorch.
        *   Trainers for these models.
        *   Agents implementing action selection, memory, and training loops.
        *   A base environment class structure.
*   **File I/O (`file_io.py`)**:
    *   Reads and writes `SurfaceData`, `SurfaceObjData`, and `SurfaceImgData` from/to CSV-like files.
*   **Data Types (`data_types.py`, `ovo_types.py`)**:
    *   Defines various `dataclass` structures used throughout the project for consistent data handling (e.g., `SurfaceData`, `Detection`, `CameraParams`, `PosAngle`).

## Requirements

### Python Requirements

*   Python 3.9+
*   See `requirements.txt` for a full list of Python package dependencies. Key dependencies include:
    *   `numpy`
    *   `opencv-python`
    *   `torch` & `torchvision`
    *   `onnxruntime` (or `onnxruntime-gpu` for CUDA support)
    *   And others as listed in `requirements.txt`.

**To install Python requirements:**
```bash
pip install -r requirements.txt
```
*(Note, that for CUDA support, you should install appropriate `pytorch` version)*

### Python Usage

* To build .pyd:
  * Install python `nuitka` library.
  * Go to the rood directory for Open_VO project
  * Run `python -m nuitka --module Open_VO --output-dir=dist`.
* To build .whl run `python setup.py bdist_wheel`.


* To use .pyd use `import Open_VO` in python scripts.
* To install .whl run 'pip install <.whl>'

### CPP Usage

* A C11 compatible C++ Compiler:
    * Windows:
      * MinGW-w64 (often installed via MSYS2 or Nuitka can prompt for it). 
      * Microsoft Visual C++ (MSVC) from Visual Studio Build Tools (e.g., VS 2019, VS 2022). 
    * Linux: GCC (e.g., g++ version 9 or newer).

For cpp usage make sure that you adjust your environment with installed OpenCV and Torch packages with dependencies:
* opencv_world4100.lib
* asmjit.lib
* c10.lib
* cpuinfo.lib
* dnnl.lib
* fbgemm.lib
* fmt.lib
* kineto.lib
* libprotobuf.lib
* libprotoc.lib
* pthreadpool.lib
* sleef.lib
* torch.lib
* torch_cpu.lib
* XNNPACK.lib

-----------------------------------------------

### This library has been created with FASIE support. Contract number "48ГУКодИИС13-D7/94521" ###