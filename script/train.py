import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader
import wandb

from config import *
from dataset import train_dataset
from model.diffusion import forward_diffusion
from model.unet import UNet

EPOCH = 50
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 128))
LEARNING_RATE = 0.0005
WANDB_PROJECT = "diffusion-morphomnist"

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

try:
    model = torch.load("model.pt", map_location=DEVICE)
    model = model.to(DEVICE)
except Exception:
    model = UNet(1).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

def init_wandb(model: nn.Module):
    run = wandb.init(
        project=WANDB_PROJECT,
        config={
            "epochs": EPOCH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "image_size": IMG_SIZE,
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
                batch_t = torch.randint(0, T, (batch_x.size(0),)).to(DEVICE)

                with autocast("cuda", enabled=USE_AMP):
                    batch_x_t, batch_noise_t = forward_diffusion(batch_x, batch_t)
                    batch_predict_t = model(batch_x_t, batch_t, batch_cls, batch_slant)
                    loss = loss_fn(batch_predict_t, batch_noise_t)

                optimizer.zero_grad(set_to_none=True)
                if USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                last_loss = loss.item()
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
            wandb.log({"epoch": epoch, "epoch_loss": last_loss}, step=n_iter)
            torch.save(model, "model.pt.tmp")
            os.replace("model.pt.tmp", "model.pt")
    finally:
        wandb.finish()
