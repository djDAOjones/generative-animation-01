from diffusers import StableDiffusionXLPipeline, UniPCMultistepScheduler
import torch
import os
import warnings

# Suppress only LoRA key warnings
warnings.filterwarnings("ignore", message=".*LoRA.*")

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

# Prompt the user for the main subject
subject = input("Enter the subject for your pixel art (e.g. 'orange cat'): ").strip()
pixel_keywords = "pixel art {} icon, bold, centered, simple, no background, game sprite, 8-bit, close-up, minimalist, 64x64 resolution".format(subject)
full_prompt = pixel_keywords
image = pipe(full_prompt, height=1024, width=1024).images[0]

# Save the image
filename = f"exports/{subject.replace(' ', '_')}_icon_bold_1024.png"
image.save(filename)
print(f"Image saved as {filename}")
