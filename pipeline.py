import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
from torch.optim import lr_scheduler

 
from torchvision.models import resnet50
from models.vit import VitGenerator 
#import timm
from torchvision.models.resnet import ResNet50_Weights

from PIL import Image
import pandas as pd
import os


from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import wandb
import sys
import time


sys.path.append('../') #"../../" for outer
from utils_dir.utils import set_all_seed,load_config
sys.path.append('/workspace/august/ToMe/')
from tome import patch

class CustomDataset(Dataset):
    def __init__(self, df:pd.DataFrame, transform:torchvision.transforms=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name =self.df.iloc[index, 0]
        image = Image.open(img_name).convert('RGB')
        label = self.df.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

class ImageClassifier:
    def __init__(self, train_df:pd.DataFrame, test_df:pd.DataFrame,log_path:str="/Users/kunkerdthaisong/cils/flooded_road/training_logs/"):
        self.train_df = train_df
        self.test_df = test_df
        self.log_path=log_path
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])

    def create_datasets(self)->Dataset:
        train_dataset = CustomDataset(df=self.train_df, transform=self.transform)
        test_dataset = CustomDataset(df=self.test_df, transform=self.transform)
        return train_dataset, test_dataset

    def create_dataloaders(self, batch_size:int=12)->DataLoader:
        train_dataset, test_dataset = self.create_datasets()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def create_model(self,finetune:bool):
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc=torch.nn.Linear(num_ftrs,3)#len(self.train_df['class'].unique())
        #model=VitGenerator("vit_base",patch_size=8,device='cpu',evaluate=False,verbose=True)
        #model.model.head = torch.nn.Linear(model.model.embed_dim,len(self.train_df['class'].unique()))
        #model=model.model
        #model = timm.create_model("vit_base_patch16_224", pretrained=True)
        #patch.timm(model)
        #model.r = 16
        #model.head=torch.nn.Linear(in_features=768, out_features=len(self.train_df["class"].unique()), bias=True)
        if finetune:
            for p in model.parameters():
                p.requires_grad=False
            for p in model.fc.parameters():
                p.requires_grad=True

        print("check layers")
        for name, param in model.named_parameters():
            print(f"{name}:   requires_grad={param.requires_grad}")
        
        return model
    
    def plot_jaa(self,plot_what:str=None)->None:
        self.current_log
        if plot_what=="loss":
            arr=np.load(self.current_log+"loss_hist.npy") #array[1,2,2,1]
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(range(1,len(arr)+1),arr, label='traning loss', marker='o')
            plt.title('traning')
            plt.xlabel('Epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig(self.current_log+"traning_loss.jpg")
            plt.close()

        elif plot_what=="acc":
            arr=np.load(self.current_log+"acc_hist.npy")
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(range(1,len(arr)+1),arr, label='validating acc', marker='o')
            plt.title('validating')
            plt.xlabel('Epochs')
            plt.ylabel('acc')
            plt.legend()
            plt.savefig(self.current_log+"validating_acc.jpg")
            plt.close()
            
    def unfreeze(self,epoch:int,model:torch.nn.Module,model_name:str="resnet50")->torch.nn.Module:
        if model_name=="resnet50":
            if epoch ==1:
                print("unfreeze layer1")
                for name, p in model.layer1.named_parameters():
                    p.requires_grad=True
                    print(f"{name}:   requires_grad={p.requires_grad}")
                return model
                
            elif epoch==8:
                print("unfreeze layer2")
                for name, p in model.layer2.named_parameters():
                    p.requires_grad=True
                    print(f"{name}:   requires_grad={p.requires_grad}")
                return model
                
            elif epoch==12:
                print("unfreeze layer3")
                for name, p in model.layer3.named_parameters():
                    p.requires_grad=True
                    print(f"{name}:   requires_grad={p.requires_grad}")
                return model
                
            elif epoch==15:
                print("unfreeze layer4")
                for name, p in model.layer4.named_parameters():
                    p.requires_grad=True
                    print(f"{name}:   requires_grad={p.requires_grad}")
                return model
                
            elif epoch==18: #unfreeze all
                print("unfreeze all")
                for name, p in model.named_parameters():
                    p.requires_grad=True
                    print(f"{name}:   requires_grad={p.requires_grad}")
            return model
        else:
            print(f"no model_name: {model_name}")

    def train_model(self, model, train_loader:DataLoader, test_loader:DataLoader, criterion:torch.nn, optimizer:torch.optim, num_epochs:int=15,lr_scheduler=None,unfreeze_each_e:bool=False):
        self.current_log=""
        #make dir in traning_logs every running
        count_log=len(os.listdir(self.log_path)) #int
        if(count_log==0):
            os.makedirs(f"{self.log_path}/1")
            self.current_log=self.log_path+"/1/"
        else:
            os.makedirs(f"{self.log_path}/"+str(count_log+1))
            self.current_log=self.log_path+"/"+str(count_log+1)+"/"
            
        max_acc=0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        l_acc=[]
        l_train_loss=[]
        for epoch in tqdm(range(num_epochs)):
            model.train()
            running_loss = 0.0
            if unfreeze_each_e:
                model=self.unfreeze(epoch=epoch,model=model,model_name="resnet50")
            
            #set lr ==0.01 at epoch0 (warm-up)
            #if epoch==0:
            #    optimizer.param_groups[0]["lr"]=0.01
            #set lr ==0.001 for normal training
            #elif epoch==1:
            #    optimizer.param_groups[0]["lr"]=0.001
                
            for inputs, labels in train_loader:
                
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                
            #lr step 
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]
            
            #for logging
            if epoch==0:
                before_lr = 0.01
                after_lr = 0.001
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f},Lr :{before_lr}/{after_lr}")
            l_train_loss.append(epoch_loss)
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = correct / total
            wandb.log({"acc_on_validation": accuracy, "train_loss": epoch_loss,"lr_before":before_lr,"lr_after":after_lr})
            print(f"Accuracy on test set: {accuracy:.4f}")
            if accuracy>=max_acc:
                max_acc=accuracy
                if epoch>=10: ### change later
                    try:
                        torch.save(model.state_dict(), self.current_log+"exp03_best.pt")
                        print("save complete")
                    except Exception as e:
                        print(e)
            l_acc.append(accuracy)
            

        print('Training completed')
        print("try save model")
        try:
            torch.save(model.state_dict(), self.current_log+"exp02_last.pt")
            print("save complete")
        except Exception as e:
            print(e)
        
        np.save(self.current_log+"loss_hist.npy",np.asarray(l_train_loss)) #save loss hist
        np.save(self.current_log+"acc_hist.npy",np.asarray(l_acc)) #save acc on valid hist

        self.plot_jaa("loss")
        self.plot_jaa("acc")

        
    def predict(self,model,test_loader:DataLoader)->np.ndarray:
        pred=[]
        model.to('cpu')
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs, labels
                outputs = model(inputs.to('cpu'))
                _, predicted = torch.max(outputs, 1)
                predicted=predicted.cpu()
                pred.append(np.asarray(predicted.squeeze(0)))
        n=0
        x=pred[0]
        while n<len(pred)-1:
            x=np.concatenate((x,pred[n+1]), axis=None)
            n+=1
        return x


if __name__ == "__main__":
    
    a=set_all_seed(42)
    a.set_seed()
    p_dir=os.getcwd()
    print("present dir is :",p_dir)
    s=time.time()
    print("-"*20+" init wandb "+"-"*20)
    wandb.init(
    # set the wandb project where this run will be logged
    project="flooded_road",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "resnet50_imnet1k",
    "dataset": "flooded_3class",
    "epochs": 30,
    "optimizer": "Adam",
    "batch_size":12,
    "device":"gpu",
    "finetune":False,
    }
    )
    e=time.time()
    print("-"*20+" init complete "+"-"*20)
    print(f"take {e-s} secconds")
    train_df = pd.read_csv("/workspace/august/flooded_road/train_3c.csv")
    val_df=pd.read_csv("/workspace/august/flooded_road/val_3c.csv")
    
    #test_df = pd.read_csv("/workspace/august/flooded_road/test_2c_water_no.csv")
    classifier = ImageClassifier(train_df, val_df,log_path=p_dir+"/"+"/training_logs/")
    train_loader, test_loader = classifier.create_dataloaders(batch_size=12) #i use test_loader
    model = classifier.create_model(finetune=False)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler=lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
    s=time.time()
    classifier.train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=30,lr_scheduler=scheduler,unfreeze_each_e=False)
    e=time.time()
    print(f"take time {e-s}")
    wandb.finish()
    print(classifier.predict(model,test_loader)) #test
    print("-"*20+" finish "+"\U0001F925"*5)