import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import os

def decode_segmentation(mask, num_classes=8):
    colors = [
        (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)
    ]
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(num_classes):
        color_mask[mask == cls] = colors[cls]
    return color_mask

def save_prediction_visual(image, gt_mask, pred_mask, idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(TF.to_pil_image(image.cpu()))
    axs[0].set_title("Original Image")

    axs[1].imshow(decode_segmentation(gt_mask))
    axs[1].set_title("Ground Truth")

    axs[2].imshow(decode_segmentation(pred_mask))
    axs[2].set_title("Prediction")

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"prediction_{idx:03d}.png"))
    plt.close()
