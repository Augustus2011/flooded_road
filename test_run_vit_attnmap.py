import torch
import numpy as np
from models import vit
from models.utils.preprocess import visualize_predict,transform
from PIL import Image

def Main():
    device='cuda' if torch.cuda.is_available() else torch.device("cpu")
    if device=='cuda':
        torch.cuda.set_device(0)
    name_model='vit_small'
    patch_size=16
    model=vit.VitGenerator(name_model=name_model,patch_size=patch_size,device=device,evaluate=True,random=False,verbose=True)
    img=Image.open("/Users/kunkerdthaisong/cils/flooded_road/1516_1.jpg")
    visualize_predict(model=model,img=img,img_size=224,patch_size=16,device='cpu')
if __name__=="__main__":
    Main()