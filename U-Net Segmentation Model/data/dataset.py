import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from torchvision.transforms.functional import resize

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None, image_ext="jpg", mask_ext="png"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.transform = transform
        self.mask_transform = mask_transform

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(image_ext)])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        mask_name = image_name.replace(f".{self.image_ext}", f"_mask.{self.mask_ext}")
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = resize(mask, image.size()[1:], interpolation=Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask
