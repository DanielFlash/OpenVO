from setuptools import setup, find_packages
import os

# Function to read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name="Open_VO",
    version="1.0.0",
    author="OOO ORIS",
    author_email="kdsflash@gmail.com",
    description="Open library for Visual Odometry, Visual Navigation with Image Processing and Reinforcement Learning tools.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    license="GNU",

    packages=find_packages(where="."),  # Looks for packages in the current directory
    package_dir={"": "."},  # Specifies that packages are in the current directory relative to setup.py

    install_requires=[
        "numpy>=2.00",
        "opencv-python>=4.10", # For cv2
        "torch>=2.7",         # For PyTorch models
        "torchvision>=0.22",   # For torchvision
        "onnxruntime>=1.22",   # For ONNX models
    ],

    # Python version requirement
    python_requires='>=3.9',

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
