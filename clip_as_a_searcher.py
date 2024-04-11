import pandas as pd
from transformers import CLIPProcessor, CLIPModel
import glob
from read_img import read_from_url
import torch
from PIL import Image

print(read_from_url("https://storage.googleapis.com/traffy_public_bucket/attachment/2022-06/182f43ac1d9c4a09d1ed8009c851f4c5f4460d3d.jpg")) #https://storage.googleapis.com/traffy_public_bucket/attachment/2022-06/182f43ac1d9c4a09d1ed8009c851f4c5f4460d3d.jpg


def run(image:Image)->torch.Tensor:
        inputs =processor(
            text=["manhole","manhole cover","sink hole","road","not road"], #text=["flood", "flooded road", "water on floor", "road", "no water","dried road"],
            images=image,
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            outputs =model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        return probs



if __name__ =="__main__":
    df=pd.read_csv("/Users/kunkerdthaisong/cils/bangkok_traffy.csv")
    df=df.dropna(subset=["type"])
    df=df[df["type"].str.contains("ท่อ")] #30288 rows

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    count=0 #handle error manually  when url is nan
    for i in df["photo"][0:]:
        img=read_from_url(i)
        probs=run(img)
        if probs.squeeze(0)[:3].sum()>0.5 and probs.squeeze(0)[-1:].sum()<0.2: #probs.squeeze(0)[1] >0.3 and probs.squeeze(0)[:3].sum()>0.55 and probs.squeeze(0)[-2:].sum()<0.25:
            img.save(f"/Users/kunkerdthaisong/cils/images2/{count}.jpg")
        count+=1



