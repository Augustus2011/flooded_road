import torch
import numpy as np
from models import vit
from models.utils.preprocess import visualize_predict,transform
from PIL import Image
import pandas as pd

def Main():
    device='cuda' if torch.cuda.is_available() else torch.device("cpu")
    if device=='cuda':
        torch.cuda.set_device(0)
    name_model='vit_large'
    patch_size=16
    model=vit.VitGenerator(name_model=name_model,patch_size=patch_size,device=device,evaluate=False,random=False,verbose=True)
    df=pd.read_csv("/Users/kunkerdthaisong/cils/test_3level.csv")
    file_name=df["path"][1]
    #2022-3zmky6_c2, 2022-hzyxqb_c1
    img=Image.open(file_name)
    visualize_predict(model=model,img_name=file_name,img=img,patch_size=patch_size,img_size=224,device='cpu')
    #model.model.head=torch.nn.Linear(model.model.embed_dim,2)
    #print(model.model.head)
    #out=model(transform(img,224).unsqueeze(0))
    #print(out.shape)
    #print(out.softmax(1))
if __name__=="__main__":
    Main()