import torch
import numpy as np
from src.segment_anything import build_sam_vit_h, SamPredictor
from src.lora import LoRA_sam
from PIL import Image
import matplotlib.pyplot as plt
import src.utils as utils
import yaml
import json


# Load SAM model and LoRA weights
sam_checkpoint = "/content/drive/MyDrive/sam_vit_h_4b8939.pth"  # Update to your checkpoint path
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = build_sam_vit_h(checkpoint=sam_checkpoint)
rank = 8  # Adjust based on your setup
sam_lora = LoRA_sam(sam, rank)
sam_lora.load_lora_parameters(f"./lora_weights/lora_rank{rank}.safetensors")
model = sam_lora.sam


def calculate_center_point(mask: np.array) -> list:
    """
    Calculate the center point of the binary mask.

    Args:
        mask (np.array): Ground truth mask.

    Returns:
        list: Single-point prompt as [[x, y]].
    """
    y_coords, x_coords = np.where(mask > 0)
    center_y, center_x = int(np.mean(y_coords)), int(np.mean(x_coords))
    return [[center_x, center_y]]


def inference_model(sam_model, image_path, filename, mask_path=None, is_baseline=False):
    """
    Perform inference with the SAM model using a single-point prompt.

    Args:
        sam_model: SAM model with or without LoRA.
        image_path (str): Path to the input image.
        filename (str): Filename for saving plots.
        mask_path (str): Path to the ground truth mask.
        is_baseline (bool): Whether to use the baseline model.
    """
    if is_baseline:
        model = build_sam_vit_h(checkpoint=sam_checkpoint)
    else:
        model = sam_model.sam

    model.eval()
    model.to(device)

    # Load image and mask
    image = Image.open(image_path)
    if mask_path:
        mask = Image.open(mask_path).convert('1')
        ground_truth_mask = np.array(mask)
        point_prompt = calculate_center_point(ground_truth_mask)
    else:
        raise ValueError("Mask path is required for inference.")

    # Predict masks using SAM
    predictor = SamPredictor(model)
    predictor.set_image(np.array(image))
    masks, iou_pred, low_res_iou = predictor.predict(
        point_coords=np.array(point_prompt),  # Single-point prompt
        point_labels=np.array([1]),  # Positive point
        multimask_output=False,
    )

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 15))
    ax1.imshow(image)
    ax1.set_title(f"Original image: {filename}")

    ax2.imshow(ground_truth_mask)
    ax2.set_title(f"Ground truth mask: {filename}")

    ax3.imshow(masks[0])
    if is_baseline:
        ax3.set_title(f"Baseline SAM prediction: {filename}")
        plt.savefig(f"./plots/{filename}_baseline.jpg")
    else:
        ax3.set_title(f"SAM LoRA rank {rank} prediction: {filename}")
        plt.savefig(f"./plots/{filename[:-4]}_rank{rank}.jpg")


# Load configuration
with open("./config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Load annotations
with open("annotations.json", "r") as f:
    annotations = json.load(f)

train_set = annotations["train"]
test_set = annotations["test"]
inference_train = True  # Set to False for test set

if inference_train:
    for image_name, dict_annot in train_set.items():
        image_path = f"./dataset/train/images/{image_name}"
        mask_path = dict_annot["mask_path"]  # Use mask_path only
        inference_model(sam_lora, image_path, filename=image_name, mask_path=mask_path, is_baseline=False)
else:
    for image_name, dict_annot in test_set.items():
        image_path = f"./dataset/test/images/{image_name}"
        mask_path = dict_annot["mask_path"]  # Use mask_path only
        inference_model(sam_lora, image_path, filename=image_name, mask_path=mask_path, is_baseline=False)
