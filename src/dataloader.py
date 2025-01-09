import torch
import glob
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class DatasetSegmentation(Dataset):
    """
    Dataset to process the images and masks with single-point prompts.

    Arguments:
        config_file (dict): The configuration dictionary with dataset paths.
        processor (Samprocessor): Samprocessor class that helps pre-process the image and prompt.
        mode (str): Either "train" or "test".
    """

    def __init__(self, config_file: dict, processor, mode: str):
        super().__init__()
        self.processor = processor

        if mode == "train":
            self.img_files = glob.glob(os.path.join(config_file["DATASET"]["TRAIN_PATH"], 'images', '*.jpg'))
            self.mask_files = []

            for img_path in self.img_files:
                # Update the mapping logic for masks
                base_name = os.path.basename(img_path).replace("screenshot_", "binary_mask_").replace(".jpg", ".png")
                mask_path = os.path.join(config_file["DATASET"]["TRAIN_PATH"], 'masks', base_name)
                self.mask_files.append(mask_path)

            # Check for missing masks
            missing_masks = [mask for mask in self.mask_files if not os.path.exists(mask)]
            if missing_masks:
                raise FileNotFoundError(f"The following masks are missing: {missing_masks}")

        else:  # For "test" mode
            self.img_files = glob.glob(os.path.join(config_file["DATASET"]["TEST_PATH"], 'images', '*.jpg'))
            self.mask_files = []

            for img_path in self.img_files:
                base_name = os.path.basename(img_path).replace("screenshot_", "binary_mask_").replace(".jpg", ".png")
                mask_path = os.path.join(config_file["DATASET"]["TEST_PATH"], 'masks', base_name)
                self.mask_files.append(mask_path)

            # Check for missing masks (optional for test mode)
            missing_masks = [mask for mask in self.mask_files if not os.path.exists(mask)]
            if missing_masks:
                raise FileNotFoundError(f"The following test masks are missing: {missing_masks}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index: int) -> dict:
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        # Load image and mask
        image = Image.open(img_path)
        mask = Image.open(mask_path).convert('1')  # Ensure binary mask

        ground_truth_mask = np.array(mask)
        original_size = tuple(image.size)[::-1]  # (H, W)

        # Calculate single-point prompt (center of the mask)
        point = self.calculate_center_point(ground_truth_mask)

        # Process data
        inputs = self.processor(image, original_size, point)
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask)

        return inputs

    @staticmethod
    def calculate_center_point(mask: np.array) -> list:
        """
        Calculate the center point of the binary mask.

        Args:
            mask (np.array): Ground truth binary mask.

        Returns:
            list: Single-point prompt as [[x, y]].
        """
        y_coords, x_coords = np.where(mask > 0)
        center_y, center_x = int(np.mean(y_coords)), int(np.mean(x_coords))
        return [[center_x, center_y]]  # Single-point format


def collate_fn(batch: list) -> list:
    """
    Used to get a list of dicts as output when using a dataloader.

    Arguments:
        batch: The batched dataset.

    Returns:
        list: List of batched dataset dictionaries.
    """
    return list(batch)
