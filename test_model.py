import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import os

class KeypointCNN(nn.Module):
    def __init__(self):
        super(KeypointCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 64 * 53, 1024)
        self.fc2 = nn.Linear(1024, 10)
        
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = KeypointCNN()
model.load_state_dict(torch.load('keypointCnnModel_small.pth'))
model.to(device)
model.eval()

def predict_keypoints(image_path, model):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image {image_path} not found.")
    
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    input_tensor = torch.tensor(normalized_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        keypoints_2d = model(input_tensor).cpu().numpy().reshape(-1, 2)
    
    return keypoints_2d

keypoint_names = ['hips', 'kneeRight', 'ankleRight', 'kneeLeft', 'ankleLeft']

def display_keypoints(image_path, keypoints, keypoint_names):
    image = cv2.imread(image_path)
    for i, keypoint in enumerate(keypoints):
        x, y = int(keypoint[0]), int(keypoint[1])
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(image, keypoint_names[i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Keypoints', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

test_image_path = 'depth_images/00067179.png'
predicted_keypoints = predict_keypoints(test_image_path, model)

display_keypoints(test_image_path, predicted_keypoints, keypoint_names)
print("Prédictions des points clés :", {name: point for name, point in zip(keypoint_names, predicted_keypoints)})