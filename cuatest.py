import torch

if torch.cuda.is_available():
    print("GPU is available!")
    print("Device name:", torch.cuda.get_device_name(0))
    print("Number of GPUs available:", torch.cuda.device_count())
    print("Current device index:", torch.cuda.current_device())
    # You can also check CUDA version if needed
    # print("CUDA version:", torch.version.cuda)
else:
    print("GPU is NOT available. PyTorch will use the CPU.")