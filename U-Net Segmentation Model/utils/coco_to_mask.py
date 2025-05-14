import os
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

def coco_segmentation_to_masks(json_path, images_dir, output_mask_dir, output_mask_format='png', image_ext='jpg'):
    os.makedirs(output_mask_dir, exist_ok=True)

    with open(json_path) as f:
        coco_data = json.load(f)

    image_id_to_info = {img['id']: img for img in coco_data['images']}
    masks = {
        img_id: np.zeros((info['height'], info['width']), dtype=np.uint8)
        for img_id, info in image_id_to_info.items()
    }

    print("[INFO] Converting segmentations to masks...")
    for ann in tqdm(coco_data['annotations']):
        img_id = ann['image_id']
        category_id = ann['category_id']
        segmentation = ann['segmentation']
        if not segmentation:
            continue

        mask_img = Image.fromarray(masks[img_id])
        draw = ImageDraw.Draw(mask_img)

        for seg in segmentation:
            if len(seg) < 6:
                continue
            polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
            draw.polygon(polygon, fill=category_id)

        masks[img_id] = np.array(mask_img)

    for img_id, mask_array in tqdm(masks.items(), desc="Saving masks"):
        image_filename = image_id_to_info[img_id]['file_name']
        mask_filename = os.path.splitext(image_filename)[0] + f"_mask.{output_mask_format}"
        mask_path = os.path.join(output_mask_dir, mask_filename)
        Image.fromarray(mask_array).save(mask_path)

    print(f"[DONE] Masks saved to {output_mask_dir}")
