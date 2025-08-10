from diffusers import StableDiffusionXLPipeline, UniPCMultistepScheduler
import torch
import os

# Load the SDXL base model
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

# Apply the LoRA weights (update path if needed)
lora_path = "pixel-art-xl.safetensors"  # LoRA file in script directory
pipe.load_lora_weights(lora_path)
pipe.fuse_lora()

# Move to best available device
if torch.backends.mps.is_available():
    pipe.to("mps")
elif torch.cuda.is_available():
    pipe.to("cuda")
else:
    pipe.to("cpu")

# Optional: swap scheduler for potentially better results
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Ensure exports folder exists
os.makedirs("exports", exist_ok=True)

# Generate an image from a text prompt
prompt = (
    "pixel art orange cat icon, bold, centered, simple, no background, game sprite, 8-bit, close-up, minimalist"
)
image = pipe(prompt, height=1024, width=1024).images[0]

# Save the image
image.save("exports/orange_cat_icon_bold_1024.png")
print("Image saved as exports/orange_cat_icon_bold_1024.png")
