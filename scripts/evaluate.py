import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import average_precision_score
from HandKeypointDetection.scripts.dataset import COCOKeypointDataset  # Import COCOKeypointDataset
from HandKeypointDetection.scripts.model import ViTPose  # Import ViTPose

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create validation dataset and dataloader
val_dataset = COCOKeypointDataset(
    root_dir='C:/Users/m2.tb/Desktop/portfolio/keypoint detection/data/val_images',
    annotation_file='C:/Users/m2.tb/Desktop/portfolio/keypoint detection/data/annotations/val.json',
    transform=transform
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = ViTPose(num_keypoints=21)  # 21 keypoints for hand
model.load_state_dict(torch.load('vitpose_hand_keypoint_coco.pth'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def compute_ap_ar(pred_keypoints, gt_keypoints, visibilities, threshold=0.5):
    valid_indices = visibilities > 0
    pred_keypoints = pred_keypoints[valid_indices]
    gt_keypoints = gt_keypoints[valid_indices]

    distances = np.sqrt(np.sum((pred_keypoints - gt_keypoints) ** 2, axis=1))

    correct = distances <= threshold
    tp = np.sum(correct)
    fp = len(correct) - tp
    fn = np.sum(visibilities == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    ap = average_precision_score(correct, -distances)

    return ap, recall

def evaluate_model(val_loader, model, device):
    model.eval()
    running_ap = 0.0
    running_ar = 0.0

    with torch.no_grad():
        for images, keypoints in val_loader:
            images = images.to(device)
            keypoints = keypoints.view(keypoints.size(0), -1).to(device)

            outputs = model(images)

            pred_keypoints = outputs.view(-1, 21, 2).cpu().numpy()
            gt_keypoints = keypoints.view(-1, 21, 2).cpu().numpy()
            visibilities = np.ones_like(gt_keypoints[:, :, 0])

            batch_ap, batch_ar = compute_ap_ar(pred_keypoints.reshape(-1, 2), gt_keypoints.reshape(-1, 2), visibilities.flatten())
            running_ap += batch_ap
            running_ar += batch_ar

    avg_ap = running_ap / len(val_loader)
    avg_ar = running_ar / len(val_loader)
    return avg_ap, avg_ar

avg_ap, avg_ar = evaluate_model(val_loader, model, device)
print(f'Average Precision (AP): {avg_ap:.4f}')
print(f'Average Recall (AR): {avg_ar:.4f}')
