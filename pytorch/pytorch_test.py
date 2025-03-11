import torch

# Check if PyTorch is installed and if CUDA is available (GPU support)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Create a tensor to test basic functionality
x = torch.rand(5, 3)
print("Random tensor:")
print(x)
