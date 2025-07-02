# OpenVO: открытая библиотека для визуальной одометрии и навигации с инструментами обработки изображений и обучением с подкреполением

Проект предоставляет коллекцию модулей на языках программирования C++ и Python для задач обработки изображений (выравнивания гистограммы, подсчет матрицы гомографии, аффинных преобразований), детекции объектов с использованием моделей в форматах ONNX и PyTorch, обработки координат, анализа карты, обработки видеопотока и основных компонентов для использования агентов в задачах обучения с подкреплением.

## Основная функциональность

*   **Обработка видео (`video_processor_ovo.py`)**:
    *   Обработка кадров видеопотока.
    *   Обнаружение ключевых точек (с использованием ORB) и подсчет дескрипторов.
    *   Расчет афинных преобразований между смежными кадрами видеопотока для отслеживания движения.
    *   Интеграция с классом `Trajectory` для обновления позиции агента, основанной на визуальной одометрии.  
*   **Обработка систем координат (`coord_calculator.py`, `ovo_types.py`, `trajectory_ovo.py`)**:
    *   Вычисление координат объектов, расположенных на карте подстилающей поверхности, с использованием детекции и географических данных.
    *   Определение общих границ карты.
    *   Вычисление координат локально найденных объектов, с использованием параметров камеры, ориентации и позиции агента.
    *   Обработка трансформаций между различными системами координат (декартова, широта / долгота).
    *   `Trajectory` класс для управления позицией и перемещением камеры / агента.  
*   **Анализ карты и коррекция навигации (`map_analysis.py`)**:
    *   Загрузка и обработка карты подстилающей поверхности (включая обработанные / необработанные данные поверхности).
    *   Сопоставление локально найденных объектов с камеры агента с объектами карты подстилающей поверхности.
    *   Расчет смещения (корректировки) используя сопоставление с шаблоном распределения объектов (картой подстилающей поверхности).
    *   Идентификация локально найденных объектов и их сопровождение между кадрами.
*   **Обработка изображений (`combined_imgproc.py`)**:
    *   Пользовательский алгоритм CLAHE (Contrast Limited Adaptive Histogram Equalization) и выравнивание гистограмм.
    *   Вычисление матрицы гомографии и афинных преобразований (линейное преобразование, метод SVD).
    *   Пользовательский алгоритм RANSAC для робастной модели.
    *   Пользовательский алгоритм KNN (K-Nearest Neighbors) для сопоставления дескрипторов.
    *   Структуры данных для описания точек, сопоставлений и дескрипторов.
*   **Детекция объектов (`detector.py`, `inferencers/`)**:
    *   Обертка для моделей детекции объектов в форматах ONNX и PyTorch.
    *   Поддержка CUDA для ускорения инференса.
    *   Также включает базовый класс инференса и его реализации для ONNX (`onnx_inferencer.py`) и PyTorch (`torch_inferencer.py`).
*   **Обучение с подкреплением (`RL_module.py`)**:
    *   Базовые реализации классов для компонентов обучения с подкреплением:
        *   Классы моделей (DQN, Policy Gradient, SARSA, A2C Actor/Value, PPO Actor/Value) с использованием PyTorch.
        *   Классы учителей для представленных моделей.
        *   Классы агентов, реализующих выбор действия, буффер памяти и цикл обучения.
        *   Базовый класс со структурой среды для обучения с подкреплением.
*   **Чтение / Запись файлов (`file_io.py`)**:
    *   Чтение и запись структур `SurfaceData`, `SurfaceObjData`, и `SurfaceImgData` из / в файлы CSV.
*   **Типы данных (`data_types.py`, `ovo_types.py`)**:
    *   Определяет различные структуры `dataclass`, используемые в других файлах проекта (например, `SurfaceData`, `Detection`, `CameraParams`, `PosAngle`).

## Требования

### Требования Python

*   Python версии 3.9+
*   Файл `requirements.txt` содержит используемые пакеты. Ключевые зависимости включают:
    *   `numpy`
    *   `opencv-python`
    *   `torch` & `torchvision`
    *   `onnxruntime` (или `onnxruntime-gpu` для поддержки CUDA)
    *   Прочие зависимости перечислены в `requirements.txt`.

**Установка зависимостей:**
```bash
pip install -r requirements.txt
```
*(Для поддержки CUDA вам необходимо установить соответствующую версию `pytorch`)*

### Использование Python (Windows 10+ / Ubuntu 22.04+)

1) Склонируйте репозиторий командой: `git clone https://github.com/DanielFlash/OpenVO.git`
2) Установите Python версии 3.9+
3) Перейдите в папку `python/source` для работы с исходным кодом или папку `python/build` для использования готовых сборок
   1) При работе на Astra Linux, установите пакет `pip` командами:
      1) `sudo apt-get upgrade`
      2) `sudo apt install python3-pip`
      3) `pip3 install --upgrade pip`
   2) Установите требуемый Python командами:
      1) `sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget`
      2) `wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz` (укажите требуемый вам Python)
      3) `tar -zxf Python-3.10.13.tgz`
      4) `cd Python-3.10.13`
      5) `./configure`
      6) `sudo make altinstall`
      7) `sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10`
      8) `sudo update-alternatives --config python3`

4) Установите зависимости командой: `pip install -r requirements.txt`
5) Соберите билд (опционально):

* Для сборки .pyd:
  * Установите библиотеку `nuitka` командой `pip install nuitka`
  * Перейдите в корневую папку проекта Open_VO
  * Запустите `python -m nuitka --module Open_VO --output-dir=dist`.
* Для сборки .whl: 
  * Запустите `python setup.py bdist_wheel` (указав соответствующий путь к файлу `setup.py`)

6) Установите и используйте библиотеку: 

* Для использования .pyd импортируйте пакет строчкой `import Open_VO` в вашем скрипте.
* Для установки .whl запустите `pip install <.whl>` (указав имя пакета), затем импортируйте пакет строчкой `import Open_VO` в вашем скрипте.

### Требования C++

* Установите компилятор С++, поддерживающий C11:
    * Для Windows:
      * MinGW-w64 
      * Microsoft Visual C++ (MSVC) из Visual Studio Build Tools (например, VS 2019, VS 2022). 
    * Для Linux: GCC (g++ версии 9+).

Для работы на С++ убедитесь, что вы настроили вашу среду разработки с установленными пакетами OpenCV и Torch со следующими зависимостями:
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

### Использование С++ (Windows 10+ / Ubuntu 22.04+)

1) Склонируйте репозиторий командой: `git clone https://github.com/DanielFlash/OpenVO.git`
2) Перейдите в папку `cpp/source` для работы с исходным кодом
3) Установите зависимости и настройте среду разработки

## Дополнительно

### Обучающие материалы и примеры использования

Вы можете изучить обучающие материалы на русском и английском языках в папке `tutorials`.

Вы можете запустить примеры использования программных модулей библиотеки на языках программирования C++ / Python в папках `cpp/examples` / `python/examples` соответственно.

### Документация

Вы можете использовать документацию для навигации между классами проекта.
Запустите `documentation/html/index.html` для просмотра в веб-браузере.

### Дообучение

Детекция объектов в разработанной библиотеке осуществляется посредством нейросетевого детектора YOLOv8. В случае необходимости, вы можете обучить свою модель YOLOv8. Для этого:
* Перейдите в папку `python/train`
* Модифицируйте конфигурационный файл `custom_seg.yaml` для использования своего датасета
* Установите зависимости: `pip install ultralitics` для использования модели YOLOv8
  * (для использования CUDA убедитесь, что установлены соответствующие версии `torch` и `torchvision`)
* Модифицируйте параметры обучения в файле `train.py`
* Запустите обучение

-----------------------------------------------

### Библиотека была разработана с поддержкой Фонда Содействия Инновациям. Номер договора "48ГУКодИИС13-D7/94521" ###

-----------------------------------------------

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

### Python Usage (Windows 10+ / Ubuntu 22.04+)

1) Clone repository: `git clone https://github.com/DanielFlash/OpenVO.git`
2) Install Python version 3.9+
3) For working with the source go to the `python/source` folder or for using built library go to the `python/build` folder
4) Install dependencies: `pip install -r requirements.txt`
5) Build your own built (optional):

* To build .pyd:
  * Install python `nuitka` library `pip install nuitka`.
  * Go to the rood directory for Open_VO project
  * Run `python -m nuitka --module Open_VO --output-dir=dist`.
* To build .whl:
  * run `python setup.py bdist_wheel`.

6) Install and use the library: 

* To use .pyd import `import Open_VO` in python scripts.
* To install .whl run `pip install <.whl>` (with the proper package name) and import `import Open_VO` in python scripts. 

### CPP Requirements

* A C11 compatible C++ Compiler:
    * Windows:
      * MinGW-w64 
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

### CPP Usage (Windows 10+ / Ubuntu 22.04+)

1) Clone repository: `git clone https://github.com/DanielFlash/OpenVO.git`
2) Go to the `cpp/source` for working with the source
3) Install dependencies and configure your environment

## Miscellaneous

### Tutorials and usage examples

You can study tutorials on russian or english languages in `tutorials` folder.

You can run usage examples on CPP / Python from `cpp/examples` / `python/examples` correspondingly.

### Documentation

You can use documentation to navigate between project classes and functionality and look for objects description.
Run `documentation/html/index.html` to open web documentation.

### Training

Developed library has object detection by using YOLOv8 neural network algorithm. You can train your own model if necessary. For this:
* Go to the `python/train` folder
* Modify configuration file `custom_seg.yaml` for custom dataset usage
* Install dependencies: `pip install ultralitics` for YOLOv8 model usage
  * (make sure you install appropriate version `torch` and `torchvision` for CUDA usage)
* Modify training parameters in the `train.py` training script
* Start training

-----------------------------------------------

### This library has been created with FASIE support. Contract number "48ГУКодИИС13-D7/94521" ###
