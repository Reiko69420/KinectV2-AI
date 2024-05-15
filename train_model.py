import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import cv2
import os

class DepthDataset(Dataset):
    def __init__(self, depth_images, keypoints, rotations):
        self.depth_images = depth_images
        self.keypoints = keypoints
        self.rotations = rotations
        
    def __len__(self):
        return len(self.depth_images)
    
    def __getitem__(self, idx):
        image_path = self.depth_images[idx]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize image
        keypoints = torch.tensor(self.keypoints[idx], dtype=torch.float32)
        rotations = torch.tensor(self.rotations[idx], dtype=torch.float32)
        return image, keypoints, rotations

class KeypointCNN(nn.Module):
    def __init__(self):
        super(KeypointCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate output size
        test_input = torch.randn(1, 1, 424, 512)
        self.feature_size = self._get_feature_size(test_input)
        
        self.fc1 = nn.Linear(self.feature_size, 1024)
        self.fc2 = nn.Linear(1024, 27)  # 27 points (5 bones * 3 vector value (x,y,z) + 4 * 3 rotation value)
        
    def _get_feature_size(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        return x.numel()

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(-1, self.feature_size)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, dataloader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        for inputs, keypoints, rotations in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.cat((keypoints, rotations), dim=1))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')
    
    torch.save(model.state_dict(), 'keypoint_cnn_model.pth')
    print("Model saved as keypoint_cnn_model.pth")

with open('results.json', 'r') as f:
    annotations = json.load(f)

depth_images = []
keypoints = []
rotations = []

for filename, data in annotations.items():
    depth_images.append(f"dataset_kinect/{filename}")
    points_3d = [data['points_3d'][point] for point in ['footLeft', 'footRight', 'kneeLeft', 'kneeRight', 'hips']]
    rotations_3d = [data['rotations_3d'][rotation] for rotation in ['footLeft', 'footRight', 'kneeLeft', 'kneeRight']]
    keypoints.append(np.array(points_3d).flatten())
    rotations.append(np.array(rotations_3d).flatten())

dataset = DepthDataset(depth_images, keypoints, rotations)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = KeypointCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, dataloader, criterion, optimizer)