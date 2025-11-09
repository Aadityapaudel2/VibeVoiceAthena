import torch, platform, sys
print("python:", sys.version.split()[0], platform.platform())
print("torch version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_runtime:", getattr(torch.version, "cuda", None))
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    print("device_name:", torch.cuda.get_device_name(0))
    print("compute_capability:", torch.cuda.get_device_capability(0))
