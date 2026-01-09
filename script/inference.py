import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
try:
    import torchdiffeq
except ImportError:
    torchdiffeq = None
import matplotlib.pyplot as plt

from config import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="cfm", choices=["cfm", "mean_flow", "mean_flow_rectified"], help="Inference method: 'cfm', 'mean_flow', or 'mean_flow_rectified'")
parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint. If None, defaults to model_{method}.pt")
args = parser.parse_args()

steps=50
method="dopri5"
rtol=1e-5
atol=1e-5

def sample(model, cls, slant):
    model.eval()
    n_sample = len(cls)
    x = torch.randn((n_sample, 1, 28, 28)).to(DEVICE)
    
    if args.method in ["mean_flow", "mean_flow_rectified"]:
        # One-step generation
        # t is irrelevant for MeanFlow model (use_time=False), but we pass a dummy
        t_dummy = torch.zeros(n_sample).to(DEVICE) 
        v = model(x, t_dummy, cls, slant)
        x = x + v
    else:
        # ODE integration for CFM
        if torchdiffeq is None:
             raise ImportError("torchdiffeq is required for CFM inference. Please install it.")
        def ode_func(t, x):
            t_expand = t.expand(x.size(0))  # [1] -> [num_samples]
            v = model(x, t_expand, cls, slant)
            return v
        ts = torch.linspace(0, 1, steps).to(DEVICE)
        x = torchdiffeq.odeint(ode_func, x, ts, method=method, rtol=rtol, atol=atol)[-1]
        
    x = (x.clamp(-1, 1) + 1) / 2 # Scale to [0, 1]
    return x.detach()

if __name__ == "__main__":
    if args.model_path is not None:
        model_name = args.model_path
    else:
        model_name = f"model_{args.method}.pt"
    try:
        model = torch.load(model_name, map_location=DEVICE, weights_only=False)
    except Exception as e:
        print(f"Model {model_name} not found or incompatible: {e}. Please train the model first.")
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
    plt.savefig(f"inference_digit_{args.method}.png")
    print(f"Saved inference_digit_{args.method}.png")
    
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
    plt.savefig(f"inference_slant_{args.method}.png")
    print(f"Saved inference_slant_{args.method}.png")

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
    plt.savefig(f"inference_both_{args.method}.png")
    print(f"Saved inference_both_{args.method}.png")
