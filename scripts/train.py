import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from torchvision import transforms
import torch.nn as nn
from transformers import ViTModel
import torch.optim as optim
from HandKeypointDetection.scripts.dataset import COCOKeypointDataset  # Import COCOKeypointDataset
from HandKeypointDetection.scripts.model import ViTPose  # Import ViTPose

class COCOKeypointDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load keypoints
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        keypoints = []
        for ann in anns:
            if 'keypoints' in ann:
                keypoints.extend(ann['keypoints'])
        keypoints = np.array(keypoints, dtype=np.float32).reshape(-1, 3)  # (num_keypoints, 3)

        # Normalize keypoints to [0, 1]
        keypoints[:, 0] /= image.shape[1]  # x-coordinate
        keypoints[:, 1] /= image.shape[0]  # y-coordinate

        # Discard visibility flags (optional)
        keypoints = keypoints[:, :2]  # Keep only (x, y)

        if self.transform:
            image = self.transform(image)

        return image, keypoints

class ViTPose(nn.Module):
    def __init__(self, num_keypoints):
        super(ViTPose, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.fc = nn.Linear(self.vit.config.hidden_size, num_keypoints * 2)  # 2 for x, y coordinates

    def forward(self, x):
        outputs = self.vit(x)
        last_hidden_state = outputs.last_hidden_state
        keypoints = self.fc(last_hidden_state[:, 0, :])  # Use the [CLS] token
        return keypoints

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloader
train_dataset = COCOKeypointDataset(
    root_dir='C:/Users/m2.tb/Desktop/portfolio/keypoint detection/data/train_images',
    annotation_file='C:/Users/m2.tb/Desktop/portfolio/keypoint detection/data/annotations/train.json',
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = ViTPose(num_keypoints=21)  # 21 keypoints for hand
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, keypoints in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# Save the model
torch.save(model.state_dict(), 'vitpose_hand_keypoint_coco.pth')
