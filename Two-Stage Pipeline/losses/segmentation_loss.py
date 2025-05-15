import torch.nn as nn
import segmentation_models_pytorch as smp

def get_loss_fn():
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = smp.losses.DiceLoss(mode="multiclass")
    return lambda pred, target: ce_loss(pred, target) + dice_loss(pred, target)
