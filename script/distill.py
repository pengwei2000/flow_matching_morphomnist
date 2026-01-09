import os
import sys
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast
try:
    import wandb
    if not hasattr(wandb, "init"):
        wandb = None
except ImportError:
    wandb = None

try:
    import torchdiffeq
except ImportError:
    torchdiffeq = None

# Add root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from script.config import *
from dataset import train_dataset
from model.dit import DiT

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="rectified_data.pt", help="Path to save/load rectified data")
parser.add_argument("--teacher_path", type=str, default="model_cfm.pt", help="Path to teacher CFM model")
parser.add_argument("--student_path", type=str, default="model_mean_flow_rectified.pt", help="Path to save student model")
parser.add_argument("--generate", action="store_true", help="Generate rectified dataset")
parser.add_argument("--train", action="store_true", help="Train student model")
parser.add_argument("--batch_size", type=int, default=400)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--resume", action="store_true", help="Resume training from student_path")
args = parser.parse_args()

def generate_rectified_data():
    if torchdiffeq is None:
        raise ImportError("torchdiffeq is required for generation.")
        
    print(f"Loading teacher model from {args.teacher_path}...")
    try:
        teacher = torch.load(args.teacher_path, map_location=DEVICE, weights_only=False)
    except Exception as e:
        # Fallback if using weights_only=True or similar issues in newer torch, 
        # though torch.load default is usually fine for full models
        print(f"Error loading model directly: {e}. Trying to load state dict if applicable or check path.")
        raise e
        
    teacher.eval()
    teacher.to(DEVICE)
    
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, # Use larger batch size for generation if VRAM allows
        num_workers=4,
        shuffle=False
    )
    
    all_x0 = []
    all_x1 = []
    all_cls = []
    all_slant = []
    
    print("Generating rectified pairs (x0 -> x1)...")
    
    # ODE parameters
    steps = 50 
    method = "dopri5"
    rtol = 1e-5
    atol = 1e-5
    
    ts = torch.linspace(0, 1, steps).to(DEVICE)
    
    with torch.no_grad():
        for i, (_, batch_cls, batch_slant) in enumerate(loader):
            batch_cls = batch_cls.to(DEVICE)
            batch_slant = batch_slant.to(DEVICE)
            n_sample = batch_cls.size(0)
            
            # Sample x0 from prior N(0, I)
            x0 = torch.randn(n_sample, 1, 28, 28).to(DEVICE)
            
            # Solve ODE to get x1
            def ode_func(t, x):
                t_expand = t.expand(x.size(0))
                return teacher(x, t_expand, batch_cls, batch_slant)
            
            # Integrate
            # We only need the final state
            # Note: odeint returns tensor of shape (n_steps, batch, ...)
            traj = torchdiffeq.odeint(ode_func, x0, ts, method=method, rtol=rtol, atol=atol)
            x1 = traj[-1]
            
            all_x0.append(x0.cpu())
            all_x1.append(x1.cpu())
            all_cls.append(batch_cls.cpu())
            all_slant.append(batch_slant.cpu())
            
            print(f"Batch {i+1}/{len(loader)} done.")
            
    # Concatenate
    data = {
        "x0": torch.cat(all_x0, dim=0),
        "x1": torch.cat(all_x1, dim=0),
        "cls": torch.cat(all_cls, dim=0),
        "slant": torch.cat(all_slant, dim=0)
    }
    
    print(f"Saving data to {args.save_path}...")
    torch.save(data, args.save_path)
    print("Done.")

def train_student():
    print(f"Loading data from {args.save_path}...")
    if not os.path.exists(args.save_path):
        print("Data file not found. Please run with --generate first.")
        return

    data = torch.load(args.save_path, weights_only=False)
    dataset = TensorDataset(data["x0"], data["x1"], data["cls"], data["slant"])
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize Student Model (MeanFlow configuration: use_time=False)
    if args.resume and os.path.exists(args.student_path):
        print(f"Resuming training from {args.student_path}...")
        student = torch.load(args.student_path, map_location=DEVICE, weights_only=False)
    else:
        print("Initializing new student model...")
        student = DiT(
            img_size=28,
            patch_size=4,
            channel=1,
            emb_size=64,
            label_num=10,
            dit_num=3,
            head=4,
            use_time=False # MeanFlow does not use time
        ).to(DEVICE)
    
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    
    scaler = torch.amp.GradScaler('cuda', enabled=(torch.cuda.is_available() and DEVICE.startswith("cuda")))
    
    print("Starting training...")
    student.train()
    
    if wandb is not None:
        wandb.init(project="diffusion-morphomnist", name="train_distill_meanflow")
        
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch_x0, batch_x1, batch_cls, batch_slant in loader:
            batch_x0 = batch_x0.to(DEVICE)
            batch_x1 = batch_x1.to(DEVICE)
            batch_cls = batch_cls.to(DEVICE)
            batch_slant = batch_slant.to(DEVICE)
            
            # Rectified Flow Loss
            # Target velocity v = x1 - x0 (assuming straight path)
            # Input to model x_t = (1-t)x0 + t*x1
            # We can sample random t for robustness, or since we enforced straightness, 
            # v(x_t) should be x1 - x0 for ALL t.
            # So we train on random t.
            
            batch_t = torch.rand(batch_x0.size(0), device=DEVICE)
            xt = (1 - batch_t.view(-1, 1, 1, 1)) * batch_x0 + batch_t.view(-1, 1, 1, 1) * batch_x1
            target_v = batch_x1 - batch_x0
            
            # IMPORTANT: Student model call
            # MeanFlow model takes (x, t, y, s) but ignores t if use_time=False
            # We pass batch_t but it should be ignored inside.
            
            with autocast('cuda', enabled=scaler.is_enabled()):
                vt = student(xt, batch_t, batch_cls, batch_slant)
                loss = loss_fn(vt, target_v)
                
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}: loss={avg_loss}")
        if wandb is not None:
            wandb.log({"epoch": epoch, "loss": avg_loss})
            
        # Save occasionally
        if (epoch + 1) % 10 == 0:
            torch.save(student, f"{args.student_path}.tmp")
            os.replace(f"{args.student_path}.tmp", args.student_path)
            
    torch.save(student, args.student_path)
    print(f"Student model saved to {args.student_path}")

if __name__ == "__main__":
    if args.generate:
        generate_rectified_data()
    if args.train:
        train_student()
