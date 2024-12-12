import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import glob
import random
import cv2


class DrowsinessDataset(Dataset):
    def __init__(self, image_dir, transform=None, validation_ratio=0.2, is_validation=False):
        self.transform = transform
        self.images = []
        self.labels = []

        
        files = list(map(lambda x: {'file': x, 'label': 1}, glob.glob( 'dataset/dataset_B_Eye_Images/openRightEyes/*.jpg')))
        files.extend(list(map(lambda x: {'file': x, 'label': 1}, glob.glob( 'dataset/dataset_B_Eye_Images/openLeftEyes/*.jpg'))))
        files.extend(list(map(lambda x: {'file': x, 'label': 0}, glob.glob( 'dataset/dataset_B_Eye_Images/closedLeftEyes/*.jpg'))))
        files.extend(list(map(lambda x: {'file': x, 'label': 0}, glob.glob('dataset/dataset_B_Eye_Images/closedRightEyes/*.jpg'))))

        
        random.shuffle(files)

        
        validation_length = int(len(files) * validation_ratio)
        if is_validation:
            files = files[:validation_length]
        else:
            files = files[validation_length:]

        
        for file in files:
            img = cv2.imread(file['file'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            self.images.append(img)
            self.labels.append(file['label'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label



transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5], std=[0.5]),  
])

train_dataset = DrowsinessDataset(
    image_dir="dataset/dataset_B_Eye_Images",
    transform=transform,
    validation_ratio=0.2,
    is_validation=False
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


val_dataset = DrowsinessDataset(
    image_dir="dataset/dataset_B_Eye_Images",
    transform=transform,
    validation_ratio=0.2,
    is_validation=True
)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"train_datase_number: {len(train_dataset)}, verify_number: {len(val_dataset)}")


model = models.mobilenet_v2(pretrained=True)
model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)  


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

for epoch in range(10):  
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

   
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}, Val Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%")


torch.save(model.state_dict(), "drowsiness_model.pth") 