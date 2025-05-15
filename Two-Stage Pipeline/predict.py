import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from config import *
from models.unet_model import get_unet_model
from utils.visualize import decode_segmentation

# Minimal Dataset for unlabeled or labeled images
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Load mask if available
        mask = None
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, image_name.replace('.jpg', '_mask.png'))
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                mask = transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST)(mask)
                mask = torch.from_numpy(np.array(mask)).long()

        return image, mask, image_name

# Run inference and visualize
def visualize_predictions(model, dataloader, class_names, save_dir="outputs/predictions"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for idx, (image, mask, name) in enumerate(tqdm(dataloader, desc="Predicting")):
            image = image.to(DEVICE)
            output = model(image)
            pred = torch.argmax(output, dim=1)[0].cpu().numpy()
            decoded_pred = decode_segmentation(pred, NUM_CLASSES)
            orig_img = TF.to_pil_image(image[0].cpu())

            fig, axs = plt.subplots(1, 3 if mask is not None else 2, figsize=(15, 5))
            axs[0].imshow(orig_img)
            axs[0].set_title("Original Image")

            if mask is not None:
                gt = mask[0].cpu().numpy()
                decoded_gt = decode_segmentation(gt, NUM_CLASSES)
                axs[1].imshow(decoded_gt)
                axs[1].set_title("Ground Truth")
                axs[2].imshow(decoded_pred)
                axs[2].set_title("Prediction")
            else:
                axs[1].imshow(decoded_pred)
                axs[1].set_title("Prediction")

            for ax in axs:
                ax.axis("off")
            plt.tight_layout()            
            filename = name if isinstance(name, str) else name[0]
            filename = os.path.splitext(filename)[0] + ".png"
            plt.savefig(os.path.join(save_dir, filename))
            plt.close()

            # Print predicted classes
            pred_classes = np.unique(pred)
            print(f"📷 {name}")
            print("✅ Predicted Classes:", [class_names[c] for c in pred_classes])
            if mask is not None:
                gt_classes = np.unique(gt)
                print("🎯 Ground Truth     :", [class_names[c] for c in gt_classes])
            print("-" * 60)

if __name__ == "__main__":
    if UNLABELED_PREDICTION:
        image_dir = UNLABELED_IMAGES_DIR
        mask_dir = None
    else:
        image_dir = TEST_IMAGES_DIR
        mask_dir = TEST_MASKS_DIR


    model = get_unet_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    dataset = InferenceDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    visualize_predictions(model, dataloader, CLASS_NAMES)
