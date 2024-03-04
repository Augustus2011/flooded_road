import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
from torchvision.models import resnet50
from models.vit import VitGenerator 
from PIL import Image
import pandas as pd
import os
from torchvision.models.resnet import ResNet50_Weights
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, df:pd.DataFrame, transform:torchvision.transforms=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name =self.df.iloc[index, 0]
        image = Image.open(img_name)
        label = int(self.df.iloc[index, 1])
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])# mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
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

    def create_model(self):
        #model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model=VitGenerator("vit_base",patch_size=8,device='cpu',evaluate=False,verbose=True)

        #num_ftrs = model.fc.in_features
        model.model.head = torch.nn.Linear(model.model.embed_dim,len(self.train_df['class'].unique()))
        return model
    
    def plot_jaa(self,plot_what:str=None):
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

    def train_model(self, model, train_loader:DataLoader, test_loader:DataLoader, criterion:torch.nn, optimizer:torch.optim, num_epochs:int=15):
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        l_acc=[]
        l_train_loss=[]
        for epoch in tqdm(range(num_epochs)):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
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
            if accuracy>max_acc:
                max_acc=accuracy
                if epoch>7: ### change later
                    try:
                        torch.save(model.state_dict(), self.current_log+"exp02_best.pt")
                        print("save complete")
                    except Exception as e:
                        print(e)
            print(f"Accuracy on test set: {accuracy:.4f}")
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
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs, labels
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                pred.append(np.asarray(predicted.squeeze(0)))
        n=0
        x=pred[0]
        while n<len(pred)-1:
            x=np.concatenate((x,pred[n+1]), axis=None)
            n+=1
        return x


if __name__ == "__main__":
    train_df = pd.read_csv("/Users/kunkerdthaisong/cils/train_3level.csv")
    test_df = pd.read_csv("/Users/kunkerdthaisong/cils/test_3level.csv")
    classifier = ImageClassifier(train_df, test_df)
    train_loader, test_loader = classifier.create_dataloaders(batch_size=16)
    model = classifier.create_model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    classifier.train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=15)
    print(classifier.predict(model,test_loader)) #test