import torch
import io
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import warnings

warnings.filterwarnings("ignore")

def transform(img:Image,image_size:int)->torch.Tensor:
    img=transforms.Resize((image_size,image_size))(img)
    img=transforms.ToTensor()(img)
    img=transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(img)# mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    return img

def visualize_predict(model,img_name:str,img:torch.Tensor,img_size:int,patch_size:int,device):
    img_pre=transform(img,img_size)
    attention=visualize_attention(model,img_pre,patch_size,device)
    plot_attention(img,attention,img_name)

def visualize_attention(model,img:torch.Tensor,patch_size:int,device):
    w, h = (
        img.shape[1] - img.shape[1] % patch_size,
        img.shape[2] - img.shape[2] % patch_size,
    )
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    attentions = model.get_last_selfattention(img.to(device))
    nh = attentions.shape[1]
    print(nh)
    print(attentions.shape)
    attentions=attentions[0,:,0,1:].reshape(nh,-1)
    attentions=attentions.reshape(nh,w_featmap,h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().detach().numpy()
    attentions=attentions.mean(axis=0,keepdims=True) #mean all attn_head
    return attentions

def plot_attention(img,attention,img_name):
    plt.figure(figsize=(10,10))
    plt.title(img_name)
    text=["img","head mean"]
    plt.subplot(1,2,1)
    plt.imshow(img.resize((224,224)))
    plt.title(text[0])
    plt.subplot(1,2,2)
    plt.imshow(attention.squeeze(0),cmap="inferno")
    plt.title(text[1])
    plt.show()
    plt.tight_layout()
    plt.axis(bool=False)