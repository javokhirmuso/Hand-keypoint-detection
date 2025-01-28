import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel



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
