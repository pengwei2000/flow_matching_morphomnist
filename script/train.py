import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader
try:
    import wandb
    if not hasattr(wandb, "init"):
        wandb = None
except ImportError:
    wandb = None

from config import *
from dataset import train_dataset
from model.dit import DiT

import argparse

EPOCH = int(os.getenv("EPOCH", 500))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 800))
LEARNING_RATE = 0.002
WANDB_PROJECT = "flow-morphomnist"

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="cfm", choices=["cfm", "mean_flow", "simple_flow"], help="Training method: 'cfm' or 'mean_flow'")
parser.add_argument("--P_mean_t", type=float, default=0.0)
parser.add_argument("--P_std_t", type=float, default=1.0)
parser.add_argument("--P_mean_r", type=float, default=0.0)
parser.add_argument("--P_std_r", type=float, default=1.0)
parser.add_argument("--ratio", type=float, default=1.0)
args = parser.parse_args()

dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    persistent_workers=True,
    shuffle=True,
    pin_memory=True,
)

torch.backends.cudnn.benchmark = True
USE_AMP = torch.cuda.is_available() and DEVICE.startswith("cuda")
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

model_name = f"model_{args.method}.pt"
use_time = True if args.method in ["cfm", "mean_flow"] else False

try:
    model = torch.load(model_name, map_location=DEVICE)
    model = model.to(DEVICE)
except Exception:
    model = DiT(
        img_size=28,
        patch_size=4,
        channel=1,
        emb_size=64,
        label_num=10,
        dit_num=3,
        head=4,
        use_time=use_time
    ).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

def logit_normal_timestep_sample(P_mean: float, P_std: float, num_samples: int, device: torch.device) -> torch.Tensor:
    rnd_normal = torch.randn((num_samples,), device=device)
    time = torch.sigmoid(rnd_normal * P_std + P_mean)
    time = torch.clip(time, min=0.0, max=1.0)
    return time

def sample_two_timesteps(num_samples, device):
    """
    Sampler (t, r): independently sample t and r, with post-processing.
    Version 1: different post-processing to ensure t >= r.
    """
    # step 1: sample two independent timesteps
    t = logit_normal_timestep_sample(-2, 2, num_samples, device=device)
    r = logit_normal_timestep_sample(-2, 2, num_samples, device=device)
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    # step 2: make t and r different with a probability of ratio 0.25.
    prob = torch.rand(num_samples, device=device)
    mask = prob < (1 - 0.25)
    r = torch.where(mask, t, r)

    return t, r

def init_wandb(model: nn.Module):
    if wandb is None:
        return None
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"train_{args.method}",
        config={
            "method": args.method,
            "epochs": EPOCH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "timesteps": T,
            "device": DEVICE,
            "amp": USE_AMP,
        },
    )
    wandb.watch(model, log="gradients", log_freq=100)
    return run


if __name__ == "__main__":
    model.train()
    n_iter = 0
    run = init_wandb(model)
    try:
        for epoch in range(EPOCH):
            last_loss = 0
            for batch_x, batch_cls, batch_slant in dataloader:
                batch_x = batch_x.to(DEVICE) * 2 - 1
                batch_cls = batch_cls.to(DEVICE)
                batch_slant = batch_slant.to(DEVICE)
                if args.method == "mean_flow":
                    # MeanFlow JVP Training
                    # Sample t, r
                    batch_t, batch_r = sample_two_timesteps(batch_x.size(0), DEVICE)
                    batch_noise = torch.randn_like(batch_x)
                    
                    z = (1 - batch_t.view(-1, 1, 1, 1)) * batch_x + batch_t.view(-1, 1, 1, 1) * batch_noise
                    v = batch_noise - batch_x

                    def u_func(z_in, t_in, r_in):
                        h_in = t_in - r_in
                        # model forward: x, t, y, s, h
                        # We need to capture batch_cls, batch_slant from outer scope
                        return model(z_in, t_in, batch_cls, batch_slant, h=h_in)

                    dtdt = torch.ones_like(batch_t)
                    drdt = torch.zeros_like(batch_r)

                    with autocast("cuda", enabled=False): # JVP often unstable or unsupported in FP16?
                        # Inputs to JVP: (z, t, r) | Tangents: (v, dtdt, drdt)
                        # We need to make sure inputs are float32 if AMP is off
                        
                        z_f32 = z.float()
                        t_f32 = batch_t.float()
                        r_f32 = batch_r.float()
                        v_f32 = v.float()
                        dtdt_f32 = dtdt.float()
                        drdt_f32 = drdt.float()


                        u_pred, dudt = torch.func.jvp(u_func, (z_f32, t_f32, r_f32), (v_f32, dtdt_f32, drdt_f32))

                        t_minus_r = (t_f32 - r_f32).view(-1, 1, 1, 1)
                        u_tgt = (v_f32 - t_minus_r * dudt).detach()
                        
                        loss = (u_pred - u_tgt)**2
                        loss = loss.sum(dim=(1, 2, 3))
                        
                        # adaptive weighting (hardcoded eps/p for now as per snippet or standard)
                        norm_eps = 1e-5
                        norm_p = 0.75 

                        adp_wt = (loss.detach() + norm_eps) ** norm_p
                        loss = loss / adp_wt
                        loss = loss.mean()

                elif args.method == "cfm":
                    # CFM Standard Training
                    batch_t = torch.rand(batch_x.size(0), device=DEVICE)
                    batch_noise = torch.randn_like(batch_x)
                    xt = (1 - batch_t.view(-1, 1, 1, 1)) * batch_noise + batch_t.view(-1, 1, 1, 1) * batch_x
                    
                    with autocast("cuda", enabled=USE_AMP):
                        vt = model(xt, batch_t, batch_cls, batch_slant)
                        loss = loss_fn(vt, batch_x - batch_noise)
                elif args.method == "simple_flow":
                    batch_t = torch.rand(batch_x.size(0), device=DEVICE)
                    batch_noise = torch.randn_like(batch_x)

                    with autocast("cuda", enabled=USE_AMP):
                        vt = model(batch_noise, batch_t, batch_cls, batch_slant)
                        loss = loss_fn(vt, batch_x - batch_noise)
                optimizer.zero_grad(set_to_none=True)
                if USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                last_loss = loss.item()
                if wandb is not None:
                    wandb.log(
                        {
                            "train/loss": last_loss,
                            "train/epoch": epoch,
                            "train/lr": optimizer.param_groups[0]["lr"],
                        },
                        step=n_iter,
                    )
                n_iter += 1

            print("epoch:{} loss={}".format(epoch, last_loss))
            if wandb is not None:
                wandb.log({"epoch": epoch, "epoch_loss": last_loss}, step=n_iter)
            torch.save(model, f"{model_name}.tmp")
            os.replace(f"{model_name}.tmp", model_name)
    finally:
        if wandb is not None:
            wandb.finish()
