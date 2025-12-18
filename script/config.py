import torch

IMG_SIZE = 48  # image resolution
T = 1000  # diffusion steps
LORA_ALPHA = 1  # LoRA scaling factor
LORA_R = 8  # LoRA rank
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
