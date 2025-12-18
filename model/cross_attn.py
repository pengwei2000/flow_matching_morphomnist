import torch
from torch import nn
from script.config import *
import math

class CrossAttention(nn.Module):
    def __init__(self,channel,qsize,vsize,fsize,cls_emb_size):
        super().__init__()
        self.w_q=nn.Linear(channel,qsize)
        self.w_k=nn.Linear(cls_emb_size,qsize)
        self.w_v=nn.Linear(cls_emb_size,vsize)
        self.softmax=nn.Softmax(dim=-1)
        self.z_linear=nn.Linear(vsize,channel)
        self.norm1=nn.LayerNorm(channel)
        self.feedforward=nn.Sequential(
            nn.Linear(channel,fsize),
            nn.ReLU(),
            nn.Linear(fsize,channel)
        )
        self.norm2=nn.LayerNorm(channel)
    
    def forward(self,x,cls_emb):
        x=x.permute(0,2,3,1)

        Q=self.w_q(x)
        Q=Q.view(Q.size(0),Q.size(1)*Q.size(2),Q.size(3))

        K=self.w_k(cls_emb)
        K=K.view(K.size(0),K.size(1),1)
        V=self.w_v(cls_emb)
        V=V.view(V.size(0),1,V.size(1))

        attn=torch.matmul(Q,K)/math.sqrt(Q.size(2))
        attn=self.softmax(attn)

        Z=torch.matmul(attn,V)
        Z=self.z_linear(Z)
        Z=Z.view(x.size(0),x.size(1),x.size(2),x.size(3))

        Z=self.norm1(Z+x)

        out=self.feedforward(Z)
        out=self.norm2(out+Z)
        return out.permute(0,3,1,2)

if __name__=='__main__':
    batch_size=2
    channel=1
    qsize=256
    cls_emb_size=32
    
    cross_atn=CrossAttention(channel=1,qsize=256,vsize=128,fsize=512,cls_emb_size=32)
    
    x=torch.randn((batch_size,channel,IMG_SIZE,IMG_SIZE))
    cls_emb=torch.randn((batch_size,cls_emb_size)) # cls_emb_size=32

    Z=cross_atn(x,cls_emb)
    print(Z.size())     # Z: (2,1,48,48)
