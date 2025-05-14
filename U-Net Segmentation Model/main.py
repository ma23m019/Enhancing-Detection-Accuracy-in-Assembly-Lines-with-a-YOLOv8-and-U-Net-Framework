
import argparse
from train import train_model
from test import evaluate_on_test
from config import *
from models.unet_model import get_unet_model
from losses.segmentation_loss import get_loss_fn
from data.dataset import SegmentationDataset
from utils.coco_to_mask import coco_segmentation_to_masks
from tuner.optuna_tuner import run_optuna_tuning

from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import os
import numpy as np

def prepare_dataloaders(batch_size):
    # Generate masks if not present
    if not os.listdir(TRAIN_MASKS_DIR):
        coco_segmentation_to_masks(TRAIN_JSON_PATH, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)
    if not os.listdir(VAL_MASKS_DIR):
        coco_segmentation_to_masks(VAL_JSON_PATH, VAL_IMAGES_DIR, VAL_MASKS_DIR)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
    ])

    train_ds = SegmentationDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, transform, mask_transform)
    val_ds = SegmentationDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR, transform, mask_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description="U-Net Segmentation Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'tune'])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_path', type=str, default=MODEL_SAVE_PATH)
    parser.add_argument('--trials', type=int, default=20, help="Only for tuning")

    args = parser.parse_args()

    if args.mode == 'train':
        print("[INFO] Starting training...")
        model = get_unet_model(NUM_CLASSES).to(DEVICE)
        loss_fn = get_loss_fn()
        train_loader, val_loader = prepare_dataloaders(args.batch_size)
        train_model(model, train_loader, val_loader, loss_fn, DEVICE, args.epochs, args.save_path)

    elif args.mode == 'test':
        print("[INFO] Running evaluation...")
        from test import run_test_evaluation
        run_test_evaluation(args.batch_size)

    elif args.mode == 'tune':
        print("[INFO] Starting hyperparameter tuning with Optuna...")
        from tuner.optuna_tuner import run_optuna_tuning
        run_optuna_tuning(trials=args.trials)
        
if __name__ == '__main__':
    main()
