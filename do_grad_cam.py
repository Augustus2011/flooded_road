import torch
import torchvision
from torchvision.transforms import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models.resnet import ResNet50_Weights
import pandas as pd


def forward_hook(module,input,output):
  activation.append(output)

def backward_hook(module,grad_in,grad_out):
  grad.append(grad_out[0])



model=torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT) #change weights to my finetuned
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs,3)
model.load_state_dict(torch.load("/Users/kunkerdthaisong/cils/flooded_road/exp02.pt"))
model.eval() #like with torch.no_grad()
target_layer=model.layer4[-1] #last layer

model.layer4[-1].register_forward_hook(forward_hook)
model.layer4[-1].register_backward_hook(backward_hook)

count=0
df=pd.read_csv("/Users/kunkerdthaisong/cils/flooded_road/test_look_grad.csv")
for i in df["path"]:
  img1=Image.open(i)
  img1=img1.resize((224,224))
  trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
  img=trans(img1)
  print(img.shape)
  img=img.unsqueeze(0)
  print(img.shape)

  # Add hook to get the tensors


  grad=[]
  activation=[]

  # forward pass to get activations
  out=model(img)

  #out[0,2] #full flood
  #out[0,1] #not full water on the road
  #out[0,0] ##pond,little water
  loss=out[0,2]

  model.zero_grad()
  loss.backward()

  grads=grad[0].cpu().data.numpy().squeeze()
  fmap=activation[0].cpu().data.numpy().squeeze()


  print("grads.shape",grads.shape)
  tmp=grads.reshape([grads.shape[0],-1])
  # Get the mean value of the gradients of every featuremap
  weights=np.mean(tmp,axis=1)
  print("weights.shape",weights.shape)


  cam = np.zeros(grads.shape[1:])
  for i,w in enumerate(weights):
    cam += w*fmap[i,:]
  cam=(cam>0)*cam
  print("cam.shape",cam.shape)
  print(cam)
  cam=cam/cam.max()*255
  print(cam)
  print(cam > 255*0.85)


  npic=np.array(img1)

  cam = cv2.resize(cam,(npic.shape[1],npic.shape[0]))
  print(cam.shape)

  heatmap=cv2.applyColorMap(np.uint8(cam),cv2.COLORMAP_JET)

  cam_img=npic*0.7+heatmap*0.3
  print(cam_img.shape)
  #display(Image.fromarray(heatmap[:,:,::-1]))
  l_img=torchvision.transforms.ToPILImage()(np.uint8(cam_img[:,:,::-1]))
  l_img.save(f"/Users/kunkerdthaisong/cils/selected_streetview/{count}.jpg")
  count+=1