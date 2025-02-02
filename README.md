# Hand Keypoint Detection

This project implements hand keypoint detection using Vision Transformer (ViT). The project includes training, evaluation, and visualization scripts.

## Project Structure

```
HandKeypointDetection/
│
├── data/
│   ├── train_images/
│   ├── val_images/
│   └── annotations/
│       ├── train.json
│       └── val.json
│
├── train.py
├── evaluate.py
├── visualize.py
└── README.md
```

## Requirements

- Python 3.10
- PyTorch
- torchvision
- transformers
- pycocotools
- OpenCV
- numpy
- matplotlib
- scikit-learn

## Training

To train the model, run:

```bash
python train.py
```

## Evaluation

To evaluate the model, run:

```bash
python evaluate.py
```

## Visualization

To visualize the keypoints, run:

```bash
python visualize.py
```
## Results



| Metric                   | Value  |
|--------------------------|----------------|
| **Loss** | 0.0027 |
| **AP** | 1.0000 |
| **AR** | 1.0000 |

## Samples
  <img src="visualizations/output.png" height="70%" width="70%"
        style="object-fit:contain"
    />

