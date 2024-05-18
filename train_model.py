import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import torch.nn as nn
import torch.optim as optim
import random

class KeypointsDataset(Dataset):
    def __init__(self, annotations_file, img_dir, num_samples=10000, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = []
        self.keypoints = []

        with open(annotations_file) as f:
            lines = f.readlines()
            i = 0
            # Select random images (Beacuse 100k is a lot, i use K2HPD/Itop dataset)
            random_lines = random.sample(lines, min(num_samples, len(lines)))
            for line in random_lines:
                parts = line.strip().split()
                img_name = parts[0]
                keypoints = list(map(int, parts[1:]))
                self.img_names.append(img_name)
                self.keypoints.append(keypoints)
                i = i + 1
                #print(i)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Image {img_path} not found.")
            return None

        keypoints = np.array(self.keypoints[idx]).reshape(-1, 2)

        if self.transform:
            image = self.transform(image)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        keypoints = keypoints.view(-1)  # Convertir en un vecteur 1D

        return image, keypoints

annotations_file = 'annotations.txt'
img_dir = 'depth_images'
dataset = KeypointsDataset(annotations_file, img_dir, num_samples=10000)

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

dataloader = DataLoader(dataset, batch_size=48, shuffle=True, collate_fn=collate_fn)

class KeypointCNN(nn.Module):
    def __init__(self):
        super(KeypointCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 64 * 53, 768)
        self.fc2 = nn.Linear(768, 10)  # 10 points (5 bones * 2 cords)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.MaxPool2d(2)(x)
        x = x.view(-1, 128 * 64 * 53)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

model = KeypointCNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, dataloader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            if inputs is None or labels is None:
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Step {i}, Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    torch.save(model.state_dict(), 'keypointCnnModel_small.pth')

train_model(model, dataloader, criterion, optimizer)