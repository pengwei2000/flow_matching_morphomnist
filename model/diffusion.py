import torch
from script.config import *
from dataset import train_dataset,tensor_to_pil
import matplotlib.pyplot as plt

betas=torch.linspace(0.0001,0.02,T)
alphas=1-betas

alphas_cumprod=torch.cumprod(alphas,dim=-1)
alphas_cumprod_prev=torch.cat((torch.tensor([1.0]),alphas_cumprod[:-1]),dim=-1)
variance=(1-alphas)*(1-alphas_cumprod_prev)/(1-alphas_cumprod)

def forward_diffusion(batch_x,batch_t):
    batch_noise_t=torch.randn_like(batch_x)
    batch_alphas_cumprod=alphas_cumprod.to(DEVICE)[batch_t].view(batch_x.size(0),1,1,1)
    batch_x_t=torch.sqrt(batch_alphas_cumprod)*batch_x+torch.sqrt(1-batch_alphas_cumprod)*batch_noise_t
    return batch_x_t,batch_noise_t

if __name__=='__main__':
    batch_x=torch.stack((train_dataset[0][0],train_dataset[1][0]),dim=0).to(DEVICE)

    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(tensor_to_pil(batch_x[0]))
    plt.subplot(1,2,2)
    plt.imshow(tensor_to_pil(batch_x[1]))
    plt.show()

    batch_x=batch_x*2-1
    batch_t=torch.randint(0,T,size=(batch_x.size(0),)).to(DEVICE)
    print('batch_t:',batch_t)
    
    batch_x_t,batch_noise_t=forward_diffusion(batch_x,batch_t)
    print('batch_x_t:',batch_x_t.size())
    print('batch_noise_t:',batch_noise_t.size())

    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(tensor_to_pil((batch_x_t[0]+1)/2))   
    plt.subplot(1,2,2)
    plt.imshow(tensor_to_pil((batch_x_t[1]+1)/2))
    plt.show()
