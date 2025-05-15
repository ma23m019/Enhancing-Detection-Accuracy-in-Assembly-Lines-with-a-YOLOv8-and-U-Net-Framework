import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from config import *
from models.unet_model import get_unet_model
from losses.segmentation_loss import get_loss_fn
from data.dataset import SegmentationDataset
from utils.metrics import compute_metrics, compute_iou, compute_detailed_metrics, plot_confusion_matrix
from utils.coco_to_mask import coco_segmentation_to_masks

# Transforms
resize_size = IMAGE_SIZE
image_transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor(),
])
mask_transform = transforms.Compose([
    transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
])

def evaluate_on_test(model, test_loader, loss_fn, device, class_names):
    model.eval()
    total_loss = 0
    all_metrics = []
    all_ious = []
    per_class_counters = {cls: {"TP": 0, "FP": 0, "FN": 0, "IoU": 0.0} for cls in range(NUM_CLASSES)}
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint64)

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="[Evaluating Test Set]"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()

            metrics = compute_metrics(outputs, masks, NUM_CLASSES)
            ious = compute_iou(outputs, masks, NUM_CLASSES)
            all_metrics.append(metrics)
            all_ious.append(ious)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks_np = masks.cpu().numpy()

            for pred, true in zip(preds, masks_np):
                for cls in range(NUM_CLASSES):
                    pred_mask = (pred == cls)
                    true_mask = (true == cls)

                    TP = (pred_mask & true_mask).sum()
                    FP = (pred_mask & ~true_mask).sum()
                    FN = (~pred_mask & true_mask).sum()
                    union = (pred_mask | true_mask).sum()
                    iou = TP / (union + 1e-6) if union > 0 else 0.0

                    per_class_counters[cls]["TP"] += TP
                    per_class_counters[cls]["FP"] += FP
                    per_class_counters[cls]["FN"] += FN
                    per_class_counters[cls]["IoU"] += iou

                # Update confusion matrix
                for t, p in zip(true.flatten(), pred.flatten()):
                    if 0 <= t < NUM_CLASSES and 0 <= p < NUM_CLASSES:
                        confusion_matrix[t, p] += 1

    # Global metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0]
    }
    mean_iou = torch.nanmean(torch.stack(all_ious)).item()

    print("\n🔍 Global Metrics:")
    print(f"Accuracy : {avg_metrics['accuracy']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall   : {avg_metrics['recall']:.4f}")
    print(f"F1-Score : {avg_metrics['f1']:.4f}")
    print(f"mIoU     : {mean_iou:.4f}")

    # Per-class table
    print("\n📊 Final Per-Class Test Metrics:")
    print(f"{'Class':<25}{'Precision':>10}{'Recall':>10}{'F1':>10}{'IoU':>10}")
    for cls in range(NUM_CLASSES):
        stats = per_class_counters[cls]
        TP, FP, FN = stats["TP"], stats["FP"], stats["FN"]
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = stats["IoU"] / len(test_loader)
        class_name = class_names[cls] if class_names else f"Class {cls}"
        print(f"{class_name:<25}{precision:10.3f}{recall:10.3f}{f1:10.3f}{iou:10.3f}")

    # Confusion Matrix Heatmap
    plot_confusion_matrix(confusion_matrix, class_names)

if __name__ == "__main__":
    # Generate masks if missing
    if not os.listdir(TEST_MASKS_DIR):
        coco_segmentation_to_masks(TEST_JSON_PATH, TEST_IMAGES_DIR, TEST_MASKS_DIR)

    # Load model
    model = get_unet_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    loss_fn = get_loss_fn()

    # Dataset & Dataloader
    test_dataset = SegmentationDataset(TEST_IMAGES_DIR, TEST_MASKS_DIR, image_transform, mask_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    evaluate_on_test(model, test_loader, loss_fn, DEVICE, CLASS_NAMES)
