#PyTorch

## Software Abstract

For my project, I decided to use PyTorch, which is an open source machine learning (ML) library used for building and training deep neural networks. It's written in Python and integrated with popular Python libraries like NumPy. PyTorch supports CPU, GPU, and parallel processing to ensure computational work can be distributed among multiple CPU and GPU cores. PyTorch is particularly useful in science and engineering for tasks involving image classification, object detection, and reinforcement learning. PyTorch is not a its own programming language, but rather alibrary that can be used in Python. It provides Python with a flexible, intuitive framework for building and training deep learning models while maintaining efficient performance.

## Installation Instructions

**Overview**
Since PyTorch is not one of the available modules on the HPCC, we have to create a conda environment to install PyTorch.

1. **Install Miniforge3** (Minimal installer for conda)
   * module purge
   * module load Miniforge3
   * module list and/or Conda --version (for verification)

2. **Create a Conda environment for PyTorch**
   * conda create --name pytorch_env python=3.10

3. **Activate the environment**
   * conda activate pytorch_env

4. **Install PyTorch**
   * conda install pytorch torchvision -c pytorch

## Example Code

The py script in the directory provides a brief verification of whether or not PyTorch was successfully installed. To run the example code:

   * Make sure you're in the Conda environment
   * Compile as usual "python pytorch_test.py"
   * Run the code

https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
https://pytorch.org/   
