##################################### Training the Model #####################################
from train import train_model, evaluate_model
from models.unet_model import get_unet_model
from losses.segmentation_loss import get_loss_fn
from data.dataset import SegmentationDataset
from utils.coco_to_mask import coco_segmentation_to_masks
from config import *

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

# Convert COCO to masks
coco_segmentation_to_masks(TRAIN_JSON_PATH, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)
coco_segmentation_to_masks(VAL_JSON_PATH, VAL_IMAGES_DIR, VAL_MASKS_DIR)

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

# Dataset + Loader
train_dataset = SegmentationDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, image_transform, mask_transform)
val_dataset = SegmentationDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR, image_transform, mask_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Train
model = get_unet_model(NUM_CLASSES).to(DEVICE)
loss_fn = get_loss_fn()
train_model(model, train_loader, val_loader, loss_fn, DEVICE, epochs=50)

##################################### Evaluating the Model #####################################
from train import train_model, evaluate_model
from models.unet_model import get_unet_model
from losses.segmentation_loss import get_loss_fn
from data.dataset import SegmentationDataset
from utils.coco_to_mask import coco_segmentation_to_masks
from config import *

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from test1 import evaluate_on_test
from models.unet_model import get_unet_model
from losses.segmentation_loss import get_loss_fn
from data.dataset import SegmentationDataset
from config import *

from torchvision import transforms
from torch.utils.data import DataLoader
import os

# Convert COCO to masks
coco_segmentation_to_masks(TEST_JSON_PATH, TEST_IMAGES_DIR, TEST_MASKS_DIR)

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

# Dataset + Loader
test_dataset = SegmentationDataset(TEST_IMAGES_DIR, TEST_MASKS_DIR, image_transform, mask_transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# Evaluate
model = get_unet_model(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
loss_fn = get_loss_fn()

evaluate_on_test(model, test_loader, loss_fn, DEVICE, CLASS_NAMES)
