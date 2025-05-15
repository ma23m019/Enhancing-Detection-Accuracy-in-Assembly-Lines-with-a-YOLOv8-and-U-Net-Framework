import os
import json
import torch
import optuna
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from config import *
from models.unet_model import get_unet_model
from losses.segmentation_loss import get_loss_fn
from data.dataset import SegmentationDataset
from utils.coco_to_mask import coco_segmentation_to_masks
from utils.logger import save_json_log, plot_metric_curve
from train import evaluate_model

# Set global device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[DEBUG] Using device: {DEVICE}")


def prepare_dataloaders(batch_size):
    # Convert COCO annotations to masks if not already done
    if not os.listdir(TRAIN_MASKS_DIR):
        coco_segmentation_to_masks(TRAIN_JSON_PATH, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)
    if not os.listdir(VAL_MASKS_DIR):
        coco_segmentation_to_masks(VAL_JSON_PATH, VAL_IMAGES_DIR, VAL_MASKS_DIR)

    image_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long()),
    ])

    train_ds = SegmentationDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, image_transform, mask_transform)
    val_ds = SegmentationDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR, image_transform, mask_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


def objective(trial):
    print("[DEBUG] Starting new Optuna trial")

    # Suggest hyperparameters
    encoder = trial.suggest_categorical("encoder_name", ["resnet18", "resnet34", "efficientnet-b0"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    use_dice = trial.suggest_categorical("use_dice", [True, False])

    # Build model
    model = get_unet_model(NUM_CLASSES)
    model.encoder_name = encoder
    model = model.to(DEVICE)
    print(f"[DEBUG] Model is on: {next(model.parameters()).device}")

    # Loss function
    if use_dice:
        from segmentation_models_pytorch.losses import DiceLoss
        ce_loss = torch.nn.CrossEntropyLoss()
        dice_loss = DiceLoss(mode="multiclass")
        loss_fn = lambda pred, target: ce_loss(pred, target) + dice_loss(pred, target)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # Data
    train_loader, val_loader = prepare_dataloaders(batch_size)

    # Optimizer
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Train loop
    for epoch in range(10):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

    # Evaluation
    val_loss, val_miou, val_metrics = evaluate_model(model, val_loader, loss_fn, DEVICE)
    
    trial.set_user_attr("val_loss", val_loss)
    trial.set_user_attr("accuracy", val_metrics.get("accuracy"))
    trial.set_user_attr("precision", val_metrics.get("precision"))
    trial.set_user_attr("recall", val_metrics.get("recall"))
    trial.set_user_attr("mIoU", val_miou)
    
    print(f"[DEBUG] Trial completed: mIoU = {val_miou:.4f}, loss = {val_loss:.4f}, accuracy = {val_metrics.get("accuracy"):.4f}, precision = {val_metrics.get("precision"):.4f}, recall = {val_metrics.get("recall"):.4f},")
    
    return val_miou


def run_optuna_tuning(trials=20):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, n_jobs=1)

    # Results
    best_trial = study.best_trial
    print("\n🎯 Best Trial:")
    print(f"mIoU: {best_trial.value:.4f}")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Save results
    os.makedirs("outputs/logs", exist_ok=True)
    save_json_log(best_trial.params, "best_params.json")
    plot_metric_curve([t.value for t in study.trials if t.value is not None], metric_name="mIoU")
    study.trials_dataframe().to_csv("outputs/logs/optuna_trials.csv", index=False)

    return best_trial.params
