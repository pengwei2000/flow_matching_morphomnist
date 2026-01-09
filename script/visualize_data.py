import torch
import matplotlib.pyplot as plt
import os
import sys
# Add root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from script.config import DEVICE

def visualize_rectified():
    save_path = "rectified_data.pt"
    if not os.path.exists(save_path):
        print(f"File {save_path} not found.")
        return

    print(f"Loading {save_path}...")
    try:
        data = torch.load(save_path, weights_only=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    x0 = data["x0"]
    x1 = data["x1"]
    cls = data["cls"]
    slant = data["slant"]
    
    print(f"Dataset size: {x0.shape[0]}")
    
    # Plot first 10 samples
    plt.figure(figsize=(10, 4))
    for i in range(10):
        # Top row: x0 (Noise)
        plt.subplot(2, 10, i+1)
        plt.imshow(x0[i].squeeze().numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f"N")
        
        # Bottom row: x1 (Generated)
        plt.subplot(2, 10, i+11)
        plt.imshow(x1[i].squeeze().numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f"{cls[i].item()}")
        
    plt.tight_layout()
    output_file = "visualize_rectified.png"
    plt.savefig(output_file)
    print(f"Saved visualization to {output_file}")

if __name__ == "__main__":
    visualize_rectified()
