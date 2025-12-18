import os
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .dataset_io import load_idx

pil_to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

tensor_to_pil = transforms.Compose(
    [
        transforms.Lambda(lambda t: t * 255),
        transforms.Lambda(lambda t: t.type(torch.uint8)),
        transforms.ToPILImage(),
    ]
)


class MorphoMNISTDataset(Dataset):
    def __init__(self, root_dir: str, train: bool = True) -> None:
        split = "train" if train else "t10k"
        self.mnist = load_idx(os.path.join(root_dir, f"{split}-images-idx3-ubyte.gz"))
        self.label = load_idx(os.path.join(root_dir, f"{split}-labels-idx1-ubyte.gz"))
        self.morpho_data = pd.read_csv(os.path.join(root_dir, f"{split}-morpho.csv"))

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        img = pil_to_tensor(transforms.ToPILImage()(self.mnist[idx]))
        label = int(self.label[idx])
        slant = torch.tensor(self.morpho_data.iloc[idx]["slant"], dtype=torch.float32)
        return img, label, slant


train_dataset = MorphoMNISTDataset(root_dir="dataset/Morpho-MNIST", train=True)


if __name__ == "__main__":
    fig, axs = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        img_tensor, label, slant = train_dataset[i]
        print(f"Label: {label}, Slant: {slant}")
        pil_img = tensor_to_pil(img_tensor)
        axs[i].imshow(pil_img, cmap="gray")
    plt.show()
