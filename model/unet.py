import torch
from torch import nn

from .time_position_emb import TimePositionEmbedding
from .conv_block import ConvBlock


class UNet(nn.Module):
    def __init__(
        self,
        img_channel: int,
        channels=None,
        time_emb_size: int = 256,
        qsize: int = 16,
        vsize: int = 16,
        fsize: int = 32,
        cls_emb_size: int = 32,
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [64, 128, 256, 512, 1024]

        channels = [img_channel] + channels

        self.time_emb = nn.Sequential(
            TimePositionEmbedding(time_emb_size),
            nn.Linear(time_emb_size, time_emb_size),
            nn.ReLU(),
        )

        self.cls_emb = nn.Embedding(10, cls_emb_size)
        self.slant_linear = nn.Linear(1, cls_emb_size)
        self.context_fusion = nn.Linear(2 * cls_emb_size, cls_emb_size)

        self.enc_convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.enc_convs.append(
                ConvBlock(
                    channels[i],
                    channels[i + 1],
                    time_emb_size,
                    qsize,
                    vsize,
                    fsize,
                    cls_emb_size,
                )
            )

        self.maxpools = nn.ModuleList()
        for _ in range(len(channels) - 2):
            self.maxpools.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.deconvs = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.deconvs.append(
                nn.ConvTranspose2d(channels[-i - 1], channels[-i - 2], kernel_size=2, stride=2)
            )

        self.dec_convs = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.dec_convs.append(
                ConvBlock(
                    channels[-i - 1],
                    channels[-i - 2],
                    time_emb_size,
                    qsize,
                    vsize,
                    fsize,
                    cls_emb_size,
                )
            )

        self.output = nn.Conv2d(channels[1], img_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cls: torch.Tensor, slant: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        cls_emb = self.cls_emb(cls)
        slant_emb = self.slant_linear(slant.unsqueeze(1))
        context = self.context_fusion(torch.cat([cls_emb, slant_emb], dim=1))

        residual = []
        for i, conv in enumerate(self.enc_convs):
            x = conv(x, t_emb, context)
            if i != len(self.enc_convs) - 1:
                residual.append(x)
                x = self.maxpools[i](x)

        for i, deconv in enumerate(self.deconvs):
            x = deconv(x)
            residual_x = residual.pop(-1)
            x = self.dec_convs[i](torch.cat((residual_x, x), dim=1), t_emb, context)

        return self.output(x)
