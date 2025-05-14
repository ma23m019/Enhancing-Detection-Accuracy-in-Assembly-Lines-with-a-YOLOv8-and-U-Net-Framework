import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_iou(preds, labels, num_classes):
    preds = torch.argmax(preds, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            ious.append(torch.tensor(float('nan')))
        else:
            ious.append(intersection / union)
    return torch.tensor(ious)


def compute_metrics(preds, labels, num_classes, exclude_bg=True):
    preds = torch.argmax(preds, dim=1)
    total_pixels = (labels != 255).sum().item()
    correct = ((preds == labels) & (labels != 255)).sum().item()
    precision_list, recall_list, f1_list = [], [], []

    class_range = range(1, num_classes) if exclude_bg else range(num_classes)

    for cls in class_range:
        pred_cls = (preds == cls)
        true_cls = (labels == cls)
        TP = (pred_cls & true_cls).sum().float()
        FP = (pred_cls & ~true_cls).sum().float()
        FN = (~pred_cls & true_cls).sum().float()
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        'accuracy': correct / total_pixels,
        'precision': torch.stack(precision_list).mean().item(),
        'recall': torch.stack(recall_list).mean().item(),
        'f1': torch.stack(f1_list).mean().item()
    }


def compute_detailed_metrics(preds, labels, num_classes):
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    per_class_metrics = {cls: {"TP": 0, "FP": 0, "FN": 0, "IoU": 0.0} for cls in range(num_classes)}

    for pred, true in zip(preds, labels):
        for cls in range(num_classes):
            pred_mask = (pred == cls)
            true_mask = (true == cls)

            TP = (pred_mask & true_mask).sum()
            FP = (pred_mask & ~true_mask).sum()
            FN = (~pred_mask & true_mask).sum()
            union = (pred_mask | true_mask).sum()
            iou = TP / (union + 1e-6) if union > 0 else 0.0

            per_class_metrics[cls]["TP"] += TP
            per_class_metrics[cls]["FP"] += FP
            per_class_metrics[cls]["FN"] += FN
            per_class_metrics[cls]["IoU"] += iou

    return per_class_metrics


def plot_confusion_matrix(conf_matrix, class_names, title="Confusion Matrix", normalize=True):
    if normalize:
        with np.errstate(all='ignore'):
            conf_matrix = conf_matrix.astype(np.float32)
            row_sums = conf_matrix.sum(axis=1, keepdims=True)
            conf_matrix = np.divide(conf_matrix, row_sums + 1e-6)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt=".2f" if normalize else "d",
                cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    plt.show()
