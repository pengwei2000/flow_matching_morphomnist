import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import matplotlib.pyplot as plt

from config import *
from model.unet import UNet
from model.diffusion import alphas, alphas_cumprod, variance

def sample(model, cls, slant):
    model.eval()
    n_sample = len(cls)
    x = torch.randn((n_sample, 1, IMG_SIZE, IMG_SIZE)).to(DEVICE)
    
    for t in reversed(range(T)):
        t_tensor = torch.full((n_sample,), t, dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            predicted_noise = model(x, t_tensor, cls, slant)
        
        alpha_t = alphas[t].to(DEVICE)
        alpha_cumprod_t = alphas_cumprod[t].to(DEVICE)
        beta_t = 1 - alpha_t
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps) + sigma * z
        x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise) + torch.sqrt(variance[t].to(DEVICE)) * noise
        
    x = (x.clamp(-1, 1) + 1) / 2 # Scale to [0, 1]
    return x

if __name__ == "__main__":
    try:
        model = torch.load("model.pt", map_location=DEVICE, weights_only=False)
    except Exception as e:
        print(f"Model not found or incompatible: {e}. Please train the model first.")
        exit()
        
    model.eval()
    
    # 1. Condition on digit (Slant = 0)
    digits = torch.arange(10).to(DEVICE)
    slants = torch.zeros(10).to(DEVICE)
    
    print("Generating digits with slant 0...")
    imgs_digit = sample(model, digits, slants)
    
    plt.figure(figsize=(10, 1))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(imgs_digit[i].cpu().squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f"{i}")
    plt.savefig("inference_digit.png")
    print("Saved inference_digit.png")
    
    # 2. Condition on slant (Digit = 5)
    digit_val = 5
    slant_vals = torch.linspace(-0.5, 0.5, 10).to(DEVICE)
    digits = torch.full((10,), digit_val, dtype=torch.long).to(DEVICE)
    
    print(f"Generating digit {digit_val} with varying slant...")
    imgs_slant = sample(model, digits, slant_vals)
    
    plt.figure(figsize=(10, 1))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(imgs_slant[i].cpu().squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f"{slant_vals[i]:.2f}")
    plt.savefig("inference_slant.png")
    print("Saved inference_slant.png")

    # 3. Condition on both (Grid)
    print("Generating grid...")
    n_digits = 10
    n_slants = 8
    slant_vals = torch.linspace(-0.5, 0.5, n_slants).to(DEVICE)
    
    plt.figure(figsize=(n_slants, n_digits))
    
    for i in range(n_digits): # Rows: Digits
        d = torch.full((n_slants,), i, dtype=torch.long).to(DEVICE)
        imgs = sample(model, d, slant_vals)
        for j in range(n_slants):
            plt.subplot(n_digits, n_slants, i * n_slants + j + 1)
            plt.imshow(imgs[j].cpu().squeeze(), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title(f"{slant_vals[j]:.1f}")
            if j == 0:
                plt.ylabel(f"{i}")
                
    plt.tight_layout()
    plt.savefig("inference_both.png")
    print("Saved inference_both.png")
