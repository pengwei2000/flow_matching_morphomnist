import torch

T = 1000  # diffusion steps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
