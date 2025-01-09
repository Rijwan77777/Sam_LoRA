import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from torch.optim import Adam
import yaml
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_h
from src.lora import LoRA_sam
import src.utils as utils

"""
This script fine-tunes the SAM ViT-H model with LoRA using single-point prompts.
The batch size, number of epochs, and other configurations are taken from the config file.
The LoRA parameters are saved at the end of training.
"""

# Load the config file
with open("./config.yaml", "r") as ymlfile:
    config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Load SAM ViT-H model
sam = build_sam_vit_h(checkpoint=config_file["SAM"]["CHECKPOINT"])

# Create SAM LoRA
sam_lora = LoRA_sam(sam, config_file["SAM"]["RANK"])
model = sam_lora.sam

# Initialize the processor and dataset
processor = Samprocessor(model)
train_dataset = DatasetSegmentation(config_file, processor, mode="train")
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config_file["TRAIN"]["BATCH_SIZE"],
    shuffle=True,
    collate_fn=collate_fn,
)

# Define optimizer and loss function
optimizer = Adam(model.image_encoder.parameters(), lr=1e-4, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

# Move model to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.train()
model.to(device)

# Training loop
num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]
for epoch in range(num_epochs):
    epoch_losses = []

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        # Forward pass with single-point prompts
        outputs = model(
            batched_input=batch,
            multimask_output=False,
        )

        # Stack ground truth and predictions for Dice loss
        stk_gt, stk_out = utils.stacking_batch(batch, outputs)
        stk_out = stk_out.squeeze(1)
        stk_gt = stk_gt.unsqueeze(1)  # Convert [H, W] to [B, C, H, W]

        # Compute loss
        loss = seg_loss(stk_out, stk_gt.float().to(device))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    # Log epoch loss
    print(f"Epoch {epoch + 1}/{num_epochs} - Mean Loss: {mean(epoch_losses):.4f}")

# Save the trained LoRA weights
rank = config_file["SAM"]["RANK"]
sam_lora.save_lora_parameters(f"lora_rank{rank}.safetensors")
