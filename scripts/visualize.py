import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from HandKeypointDetection.scripts.dataset import COCOKeypointDataset  # Import COCOKeypointDataset
from HandKeypointDetection.scripts.model import HandKeypointModel  # Import the pre-trained model

def unnormalize_image(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.permute(1, 2, 0).cpu().numpy()  # Change shape to (H, W, 3)
    image = (image * std) + mean  # Unnormalize
    image = np.clip(image, 0, 1)  # Clip to valid range
    return image

def visualize_keypoints(images, keypoints_list):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # Create a 2x2 grid of subplots
    axes = axes.ravel()  # Flatten the axes array for easy iteration

    for i, (image, keypoints) in enumerate(zip(images, keypoints_list)):
        image = unnormalize_image(image)

        ax = axes[i]
        ax.imshow(image)

        height, width, _ = image.shape
        keypoints[:, 0] *= width  # Scale x-coordinates
        keypoints[:, 1] *= height  # Scale y-coordinates

        ax.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=10)
        ax.axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()

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

# Load the pre-trained model
model = HandKeypointModel()
model.load_state_dict(torch.load('C:/Users/m2.tb/Desktop/portfolio/keypoint detection/models/pretrained_model.pth'))
model.eval()

# Get 4 images and keypoints from the dataset
images = []
keypoints_list = []
for i in range(4):
    image, _ = train_dataset[i]  # Get the i-th image
    images.append(image)
    with torch.no_grad():
        keypoints = model(image.unsqueeze(0)).squeeze(0).cpu().numpy()  # Predict keypoints
    keypoints_list.append(keypoints)

# Visualize the images and keypoints
visualize_keypoints(images, keypoints_list)
