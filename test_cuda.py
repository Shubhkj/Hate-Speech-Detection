import torch

print("="*50)
print("CUDA Test")
print("="*50)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Current Device: {torch.cuda.current_device()}")
    
    # Test tensor creation on GPU
    x = torch.randn(3, 3).cuda()
    print(f"\nTest tensor device: {x.device}")
    print("GPU test PASSED!")
else:
    print("CUDA NOT AVAILABLE - Using CPU")
print("="*50)
