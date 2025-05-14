import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from config import *
from models.unet_model import get_unet_model
from losses.segmentation_loss import get_loss_fn
from data.dataset import SegmentationDataset
from utils.metrics import compute_iou, compute_metrics, compute_detailed_metrics
from utils.coco_to_mask import coco_segmentation_to_masks

resize_size = IMAGE_SIZE
image_transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor(),
])
mask_transform = transforms.Compose([
    transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
])

def train_model(model, train_loader, val_loader, loss_fn, DEVICE, epochs=25):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_ious = []
        train_metrics_all = []

        loop = tqdm(train_loader, desc=f"[Train Epoch {epoch+1}/{epochs}]")
        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            iou = compute_iou(outputs.detach(), masks, NUM_CLASSES)
            metrics = compute_metrics(outputs.detach(), masks, NUM_CLASSES)

            train_ious.append(iou)
            train_metrics_all.append(metrics)
            loop.set_postfix(loss=loss.item(), mIoU=torch.nanmean(iou).item())

        scheduler.step()

        train_mean_iou = torch.nanmean(torch.stack(train_ious)).item()
        train_avg_metrics = {
            key: np.mean([m[key] for m in train_metrics_all])
            for key in train_metrics_all[0]
        }

        print(f"\n[Train Metrics] mIoU: {train_mean_iou:.4f} | "
              f"Acc: {train_avg_metrics['accuracy']:.4f} | "
              f"Prec: {train_avg_metrics['precision']:.4f} | "
              f"Recall: {train_avg_metrics['recall']:.4f} | "
              f"F1: {train_avg_metrics['f1']:.4f}")

        val_loss, val_mean_iou, val_avg_metrics = evaluate_model(
            model, val_loader, loss_fn, DEVICE
        )

        print(f"[Val Metrics]   Loss: {val_loss:.4f} | mIoU: {val_mean_iou:.4f} | "
              f"Acc: {val_avg_metrics['accuracy']:.4f} | "
              f"Prec: {val_avg_metrics['precision']:.4f} | "
              f"Recall: {val_avg_metrics['recall']:.4f} | "
              f"F1: {val_avg_metrics['f1']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Best model saved at epoch {epoch+1}")
            
    #<after training loop in train_model function>

    print("\n📊 Final Per-Class Validation Metrics:")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img, mask in val_loader:
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            output = model(img)
            all_preds.append(output)
            all_labels.append(mask)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    detailed_metrics = compute_detailed_metrics(all_preds, all_labels, NUM_CLASSES)

    print(f"{'Class':<25}{'Precision':>10}{'Recall':>10}{'F1':>10}{'IoU':>10}")
    for cls in range(NUM_CLASSES):
        stats = detailed_metrics[cls]
        TP, FP, FN = stats["TP"], stats["FP"], stats["FN"]
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = stats["IoU"] / len(val_loader)

        class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
        print(f"{class_name:<25}{precision:10.3f}{recall:10.3f}{f1:10.3f}{iou:10.3f}")


def evaluate_model(model, val_loader, loss_fn, DEVICE):
    model.eval()
    total_loss = 0
    all_ious = []
    all_metrics = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()

            iou = compute_iou(outputs, masks, NUM_CLASSES)
            metrics = compute_metrics(outputs, masks, NUM_CLASSES)
            all_ious.append(iou)
            all_metrics.append(metrics)

    mean_iou = torch.nanmean(torch.stack(all_ious)).item()
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0]
    }

    return total_loss / len(val_loader), mean_iou, avg_metrics

if __name__ == "__main__":
    # Convert COCO JSON to masks automatically
    coco_segmentation_to_masks(TRAIN_JSON_PATH, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)
    coco_segmentation_to_masks(VAL_JSON_PATH, VAL_IMAGES_DIR, VAL_MASKS_DIR)

    model = get_unet_model(NUM_CLASSES).to(DEVICE)
    loss_fn = get_loss_fn()

    train_dataset = SegmentationDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, image_transform, mask_transform)
    val_dataset = SegmentationDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR, image_transform, mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    train_model(model, train_loader, val_loader, loss_fn, DEVICE, epochs=25)
