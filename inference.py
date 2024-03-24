

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
#model
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models import resnet50
import timm
import time
# db and transforms and image mn
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
import PIL
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


#do quantize

import copy
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

#manage path
import glob
import sys
import os
from pathlib import Path


sys.path.append('../') #"../../" for outer of outer
from utils_dir.utils import set_all_seed,load_config
sys.path.append('/workspace/august/ToMe/')
from tome import patch ,vis


import wandb #when i want to do evaluae model perf
import psutil
#plot
import matplotlib.pyplot as plt

class Main:

    def __init__(self,vis_mode:str="vit_ToMe",path_dir:str="/workspace/august/flooded_road/train_test_flood_img/val_/",quantize:bool=False):
        self.vis_mode=vis_mode
        self.path_dir=path_dir
        self.log="/workspace/august/flooded_road/training_logs/30/"
        self.quantize=quantize
        
    def draw_and_save(self,img:PIL.Image,cls:int,path_log:str,name:str):
        draw = ImageDraw.Draw(img)
        text = str(cls)
        font_size = 36  # Set the font size
        
        font = ImageFont.load_default()  
        
        text_width, text_height = draw.textsize(text, font=font)

        position = (10, 10)
        draw.text(position, text, fill="red", font=font)
        
        img.save(path_log + name)
        
        
    def run(self):
        if self.vis_mode=="resnet50":
            
            wandb.init(
            # set the wandb project where this run will be logged
            project="flooded_road_no_water_inference",
            
            # track hyperparameters and run metadata
            config={
            "architecture": "resnet50_imnet1k",
            "dataset": "flooded_3class",
            "compress":"no",
            "batch_size":1,
            "device":"gpu",
            }
            )
            def forward_hook(module,input,output):
                activation.append(output)
        
            def backward_hook(module,grad_in,grad_out):
                grad.append(grad_out[0])
                
            model=torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT) #change to my finetuned model
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs,3)
            
            model.load_state_dict(torch.load("/workspace/august/flooded_road/training_logs/30/exp03_best.pt"))
            model.to("cpu")
            model.eval() #like with torch.no_grad()
            if self.quantize:
                example_inputs = (torch.randn(1, 3, 224, 224),)
                qconfig = get_default_qconfig("fbgemm")
                qconfig_dict = {"": qconfig}
                model_prepared=prepare_fx(model, qconfig_dict,example_inputs=example_inputs)
                calibration_data = [torch.randn(1, 3, 224, 224) for _ in range(100)]
                for i in range(len(calibration_data)):
                   model_prepared(calibration_data[i])
                model=convert_fx(copy.deepcopy(model_prepared))
                for i in glob.glob(self.path_dir+"/*.jpg"):
                    s=time.time()
                    name=os.path.basename(i)
                    img1=Image.open(i).convert('RGB')
                    img1=img1.resize((224,224))
                    trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
                    img=trans(img1)
                    img=img.unsqueeze(0)
                    out=model(img)
                    e=time.time()
                    self.draw_and_save(img=img1,cls=out.argmax(dim=-1),path_log=self.log,name=name)
                    wandb.log({"time_per_epoch":e-s})
                
            else:
                target_layer=model.layer4[-1] #last layer
                model.layer4[-1].register_forward_hook(forward_hook)
                model.layer4[-1].register_backward_hook(backward_hook)
                for i in glob.glob(self.path_dir+"/*.jpg"):
                    s=time.time()
                    name=os.path.basename(i)
                    img1=Image.open(i).convert('RGB')
                    img1=img1.resize((224,224))
                    trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
                    img=trans(img1)
                    img=img.unsqueeze(0)
                    grad=[]
                    activation=[]
                    out=model(img)
                    e=time.time()
                    
                    #out[0,0] #no water on the road 
                    #out[0,1] #little of water  
                    #out[0,2] #full of water
                    loss=out[0,int(out.argmax(dim=-1))] #this case nowater and little water
                    model.zero_grad()
                    loss.backward()
                
                    grads=grad[0].cpu().data.numpy().squeeze()
                    fmap=activation[0].cpu().data.numpy().squeeze()
                    
                    tmp=grads.reshape([grads.shape[0],-1])
                    weights=np.mean(tmp,axis=1)
                    cam = np.zeros(grads.shape[1:])
                    for i,w in enumerate(weights):
                        cam += w*fmap[i,:]
                    cam=(cam>0)*cam #cut-off
                    cam=cam/cam.max()*255
                    npic=np.array(img1)
                    cam = cv2.resize(cam,(npic.shape[1],npic.shape[0]))
                    heatmap=cv2.applyColorMap(np.uint8(cam),cv2.COLORMAP_JET)
                    cam_img=npic*0.7+heatmap*0.3
                    out_img=torchvision.transforms.ToPILImage()(np.uint8(cam_img[:,:,::-1]))
                    self.draw_and_save(img=out_img,cls=out.argmax(dim=-1),path_log=self.log,name=name)
                    out_img.save(self.log+name) #3zm
                    wandb.log({"time_per_epoch":e-s})
                
        elif self.vis_mode=="vit_ToMe":
            model = timm.create_model("vit_base_patch16_224", pretrained=True)
            patch.timm(model, trace_source=True)
            input_size = model.default_cfg["input_size"][1]
            transform_list = [
                transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(input_size)
            ]
            
            transform_vis  = transforms.Compose(transform_list)
            transform_norm = transforms.Compose(transform_list + [
                transforms.ToTensor(),
                transforms.Normalize(model.default_cfg["mean"], model.default_cfg["std"]),
            ])
            model.eval()
    
            img=Image.open("/workspace/august/flooded_road/train_test_flood_img/test/2022-HZYXQB_c1.jpg")#/workspace/august/flooded_road/train_test_flood_img/test/2022-HZYXQB_c1.jpg
            img_vis = transform_vis(img)
            img_norm = transform_norm(img)
            model.r=16
            _ = model(img_norm[None, ...])
            source = model._tome_info["source"]
            print(source,source.shape)
            out_img=vis.make_visualization(img_vis, source, patch_size=16, class_token=True)
            out_img.save("2022-HZYXQB_c1.jpg") #2022-HZYXQB_c1.jpg
        
if __name__=="__main__":
    a=set_all_seed(42)
    a.set_seed()
    main=Main(vis_mode="resnet50",quantize=False) #vis_mode="resnet50","ToMe"
    main.run()