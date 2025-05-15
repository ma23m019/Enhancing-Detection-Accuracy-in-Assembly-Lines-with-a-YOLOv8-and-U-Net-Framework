import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from utils.visualize import decode_segmentation_mask
from utils.metrics import compute_metrics, compute_detailed_metrics, plot_confusion_matrix
from models.unet import get_unet_model
from models.yolo import load_yolov8_model
from data.dataset import SegmentationDataset
from config import *

# Configuration
IMAGE_DIR = TEST_IMAGES_DIR
MASK_DIR = TEST_MASKS_DIR
CLASS_NAMES = ["BG", "Inlet Valve", "Outlet Valve", "Retainer", "SSW", "Split Collets", "Spring", "Spring Seat Washer"]
YOLO_THRESHOLD = 0.6
UNET_THRESHOLD = 0.7
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
])

dataset = SegmentationDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    transform=transform,
    mask_transform=mask_transform
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Load Models
yolo_model = load_yolov8_model("weights/yolov8_best.pt")
yolo_model.to(DEVICE).eval()

unet_model = get_unet_model(NUM_CLASSES)
unet_model.load_state_dict(torch.load("weights/unet_best.pth", map_location=DEVICE))
unet_model.to(DEVICE).eval()

# Initialize metric accumulators
all_preds = []
all_labels = []

# Inference Loop
for image, mask in tqdm(dataloader, desc="Running Two-Stage Pipeline"):
    image = image.to(DEVICE)
    mask = mask.to(DEVICE)

    # Stage 1: YOLO
    yolo_preds = yolo_model(image)
    boxes = yolo_preds[0].boxes
    valid_classes = set()
    if boxes is not None:
        for box in boxes:
            if box.conf.item() > YOLO_THRESHOLD:
                valid_classes.add(int(box.cls.item()))

    # Stage 2: U-Net
    with torch.no_grad():
        unet_out = torch.softmax(unet_model(image), dim=1)
        unet_pred = torch.argmax(unet_out, dim=1).squeeze().cpu()

    # Apply threshold-based filtering
    final_pred = torch.zeros_like(unet_pred)
    for cls in valid_classes:
        prob_map = unet_out[0, cls, :, :].cpu()
        mask_cls = (prob_map > UNET_THRESHOLD).long()
        final_pred[mask_cls == 1] = cls

    all_preds.append(final_pred)
    all_labels.append(mask.squeeze().cpu())

# Compute and Export Metrics
stacked_preds = torch.stack(all_preds)
stacked_labels = torch.stack(all_labels)
metrics = compute_metrics(stacked_preds, stacked_labels, NUM_CLASSES)
per_class_metrics, conf_matrix = compute_detailed_metrics(stacked_preds, stacked_labels, NUM_CLASSES)

# Save global metrics
df_metrics = pd.DataFrame([metrics])
df_metrics.to_csv("two_stage_metrics.csv", index=False)

# Save per-class metrics
per_class_df = pd.DataFrame(columns=["Class", "Precision", "Recall", "F1", "IoU"])
for cls in range(NUM_CLASSES):
    stats = per_class_metrics[cls]
    TP, FP, FN = stats["TP"], stats["FP"], stats["FN"]
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = stats["IoU"] / len(dataloader)
    per_class_df.loc[cls] = [CLASS_NAMES[cls], precision.item(), recall.item(), f1.item(), iou.item()]

per_class_df.to_csv("two_stage_per_class_metrics.csv", index=False)

# Print results
print("\n📊 Two-Stage Pipeline Evaluation:")
for k, v in metrics.items():
    print(f"{k.capitalize()}: {v:.4f}")

print("\n📌 Per-Class Metrics:")
print(per_class_df)

# Plot and save confusion matrix
plot_confusion_matrix(conf_matrix, CLASS_NAMES, title="Two-Stage Pipeline Confusion Matrix")
