import torch
import torch.nn as nn
import cv2
import numpy as np
import json
import os

class KeypointCNN(nn.Module):
    def __init__(self):
        super(KeypointCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        test_input = torch.randn(1, 1, 424, 512)
        self.feature_size = self._get_feature_size(test_input)
        
        self.fc1 = nn.Linear(self.feature_size, 1024)
        self.fc2 = nn.Linear(1024, 27)
        
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

def load_model(model_path):
    model = KeypointCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0 
    return image

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
    keypoints = output[:, :15].view(-1, 5, 3)
    rotations = output[:, 15:].view(-1, 4, 3)
    return keypoints, rotations

def main(image_path, model_path):
    model = load_model(model_path)
    image_tensor = preprocess_image(image_path)
    keypoints, rotations = predict(model, image_tensor)
    
    print("Keypoints (3D):", keypoints)
    print("Rotations (3D):", rotations)

if __name__ == "__main__":
    image_path = ""
    model_path = ""
    main(image_path, model_path)