from src.segment_anything.utils.transforms import ResizeLongestSide
import numpy as np
import torch
import PIL


class Samprocessor:
    """
    Processor that transforms the image and single-point prompt with ResizeLongestSide,
    then preprocesses both data for SAM input.
    """

    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def __call__(self, image: PIL.Image, original_size: tuple, point: list) -> dict:
        # Processing of the image
        image_torch = self.process_image(image, original_size)

        # Transform input single-point prompt
        point_torch = self.process_point(point, original_size)

        return {
            "image": image_torch,
            "original_size": original_size,
            "point_coords": point_torch,
            "point_labels": torch.tensor([1], dtype=torch.long, device=self.device),  # Positive point
        }

    def process_image(self, image: PIL.Image, original_size: tuple) -> torch.Tensor:
        """Preprocess the image for SAM."""
        nd_image = np.array(image)
        input_image = self.transform.apply_image(nd_image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        return input_image_torch

    def process_point(self, point: list, original_size: tuple) -> torch.Tensor:
        """Preprocess the single-point prompt for SAM."""
        point_torch = torch.tensor(point, dtype=torch.float, device=self.device).unsqueeze(0)
        return point_torch

    @property
    def device(self):
        return self.model.device

    def reset_image(self):
        """Reset the image context."""
        self.is_image_set = False
        self.features = None
