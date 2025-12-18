from torch import nn
from .cross_attn import CrossAttention

class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,time_emb_size,qsize,vsize,fsize,cls_emb_size):
        super().__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

        self.time_emb_linear=nn.Linear(time_emb_size,out_channel)
        self.relu=nn.ReLU()

        self.seq2=nn.Sequential(
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

        self.crossattn=CrossAttention(channel=out_channel,qsize=qsize,vsize=vsize,fsize=fsize,cls_emb_size=cls_emb_size)

    def forward(self,x,t_emb,cls_emb):
        x=self.seq1(x)
        t_emb=self.relu(self.time_emb_linear(t_emb)).view(x.size(0),x.size(1),1,1)
        output=self.seq2(x+t_emb)
        return self.crossattn(output,cls_emb)
