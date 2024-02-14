import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
from torchvision.models import resnet50
from PIL import Image
import pandas as pd
import os
from torchvision.models.resnet import ResNet50_Weights
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, df=pd.DataFrame(), transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name =self.df.iloc[index, 1]
        image = Image.open(img_name)
        label = int(self.df.iloc[index, 2])

        if self.transform:
            image = self.transform(image)

        return image, label


class ImageClassifier:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def create_datasets(self):
        train_dataset = CustomDataset(df=self.train_df, transform=self.transform)
        test_dataset = CustomDataset(df=self.test_df, transform=self.transform)
        return train_dataset, test_dataset

    def create_dataloaders(self, batch_size:int=6):
        train_dataset, test_dataset = self.create_datasets()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def create_model(self):
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(self.train_df['class'].unique()))
        return model

    def train_model(self, model, train_loader, test_loader, criterion, optimizer, num_epochs=15):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

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
            print(f"Accuracy on test set: {accuracy:.4f}")

        print('Training completed.')
        print("try save model")
        try:
            torch.save(model.state_dict(), "/Users/kunkerdthaisong/cils/flooded_road/exp01.pt")
            print("save complete")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    train_df = pd.read_csv("/Users/kunkerdthaisong/cils/train.csv")
    test_df = pd.read_csv("/Users/kunkerdthaisong/cils/test.csv")


    classifier = ImageClassifier(train_df, test_df)
    train_loader, test_loader = classifier.create_dataloaders(batch_size=4)
    model = classifier.create_model()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    classifier.train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=15)

