import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO

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
