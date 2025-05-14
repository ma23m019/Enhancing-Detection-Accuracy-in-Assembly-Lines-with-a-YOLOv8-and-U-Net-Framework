import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = (256, 256)
NUM_CLASSES = 8
CLASS_NAMES = [
    "BG", "Inlet Valve", "Outlet Valve", "Retainer",
    "SSW", "Split Collets", "Spring", "Spring Seat Washer"
]

TRAIN_IMAGES_DIR = "/home/priyanka/MyEnv/MyEnv/Feb-data-(segmentation)-1/train"
TRAIN_MASKS_DIR = "/home/priyanka/MyEnv/MyEnv/Feb-data-(segmentation)-1/train/masks"
VAL_IMAGES_DIR = "/home/priyanka/MyEnv/MyEnv/Feb-data-(segmentation)-1/valid"
VAL_MASKS_DIR = "/home/priyanka/MyEnv/MyEnv/Feb-data-(segmentation)-1/valid/masks"
TEST_IMAGES_DIR = "/home/priyanka/MyEnv/MyEnv/Feb-data-(segmentation)-1/test"
TEST_MASKS_DIR = "/home/priyanka/MyEnv/MyEnv/Feb-data-(segmentation)-1/test/masks"

TRAIN_JSON_PATH = "/home/priyanka/MyEnv/MyEnv/Feb-data-(segmentation)-1/train/_annotations.coco.json"
VAL_JSON_PATH = "/home/priyanka/MyEnv/MyEnv/Feb-data-(segmentation)-1/valid/_annotations.coco.json"
TEST_JSON_PATH = "/home/priyanka/MyEnv/MyEnv/Feb-data-(segmentation)-1/test/_annotations.coco.json"

UNLABELED_PREDICTION = False
UNLABELED_IMAGES_DIR = "/path/to/unlabeled/images"

MODEL_SAVE_PATH = "outputs/best_model_1704.pth"
