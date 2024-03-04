import torch
import io
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import warnings
import cv2

warnings.filterwarnings("ignore")

def transform(img:Image,image_size:int)->torch.Tensor:
    img=transforms.Resize((image_size,image_size))(img)
    img=transforms.ToTensor()(img)
    img=transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(img)# mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
    return img

def visualize_predict(model,img:torch.Tensor,img_size:int,patch_size:int,device):
    img_pre=transform(img,img_size)
    attention=visualize_attention(model,img_pre,patch_size,device)
    plot_attention(img,attention)

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

    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = (
        nn.functional.interpolate(
            attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest"
        )[0]
        .cpu()
        .numpy()
    )
    attentions=np.mean(attentions,0)

    return attentions

def plot_attention(img,attention):
    plt.figure(figsize=(10,10))
    text=["img","head mean"]
    img=img.resize((attention.shape[1],attention.shape[1]))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title(text[0])
    plt.subplot(1,2,2)
    plt.imshow(attention,cmap="inferno")
    plt.title(text[1])

    plt.show()
    plt.tight_layout()
    plt.axis(bool=False)