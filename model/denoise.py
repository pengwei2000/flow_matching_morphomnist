import torch
from script.config import *
from .diffusion import *
import matplotlib.pyplot as plt
from dataset import tensor_to_pil


def backward_denoise(model,batch_x_t,batch_cls):
    steps=[batch_x_t,]

    global alphas,alphas_cumprod,variance

    model=model.to(DEVICE)
    batch_x_t=batch_x_t.to(DEVICE)
    alphas=alphas.to(DEVICE)
    alphas_cumprod=alphas_cumprod.to(DEVICE)
    variance=variance.to(DEVICE)
    batch_cls=batch_cls.to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        for t in range(T-1,-1,-1):
            batch_t=torch.full((batch_x_t.size(0),),t).to(DEVICE) #[999,999,....]
            batch_predict_noise_t=model(batch_x_t,batch_t,batch_cls)
            shape=(batch_x_t.size(0),1,1,1)
            batch_mean_t=1/torch.sqrt(alphas[batch_t].view(*shape))*  \
                (
                    batch_x_t-
                    (1-alphas[batch_t].view(*shape))/torch.sqrt(1-alphas_cumprod[batch_t].view(*shape))*batch_predict_noise_t
                )
            if t!=0:
                batch_x_t=batch_mean_t+ \
                    torch.randn_like(batch_x_t)* \
                    torch.sqrt(variance[batch_t].view(*shape))
            else:
                batch_x_t=batch_mean_t
            batch_x_t=torch.clamp(batch_x_t, -1.0, 1.0).detach()
            steps.append(batch_x_t)
    return steps 

if __name__=='__main__':
    model=torch.load('model.pt')
    print(model)

    batch_size=10
    batch_x_t=torch.randn(size=(batch_size,1,28,28))
    batch_cls=torch.arange(start=0,end=10,dtype=torch.long)
    steps=backward_denoise(model,batch_x_t,batch_cls)
    num_imgs=20
    plt.figure(figsize=(15,15))
    for b in range(batch_size):
        for i in range(0,num_imgs):
            idx=int(T/num_imgs)*(i+1)
            final_img=(steps[idx][b].to('cpu')+1)/2
            final_img=tensor_to_pil(final_img)
            plt.subplot(batch_size,num_imgs,b*num_imgs+i+1)
            plt.imshow(final_img)
    plt.show()
