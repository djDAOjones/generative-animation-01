from diffusers import DiffusionPipeline, UniPCMultistepScheduler
import torch
import matplotlib.pyplot as plt
import math

# Load the Stable Diffusion pipeline
pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)

# Move to best available device
if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon GPU (MPS backend)")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA GPU")
else:
    device = "cpu"
    print("No GPU available, running on CPU. This will be much slower.")
pipeline.to(device)

# Use UniPCMultistepScheduler for high quality and speed
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

prompt = "An image of a squirrel in Picasso style"
num_inference_steps = 50
save_interval = 10
images = []

# Callback to capture images at intervals
def save_intermediate_images(step, timestep, callback_kwargs):
    if (step + 1) % save_interval == 0 or step == 0 or (step + 1) == num_inference_steps:
        images.append(callback_kwargs["images"][0])

# Generate image and collect intermediates
_ = pipeline(
    prompt,
    num_inference_steps=num_inference_steps,
    callback=save_intermediate_images,
    callback_steps=1
)

# Plot and save tiled grid
cols = 5
rows = math.ceil(len(images) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
for i, ax in enumerate(axes.flat):
    if i < len(images):
        ax.imshow(images[i])
        ax.set_title(f"Step {(i)*save_interval+1}")
    ax.axis('off')
plt.tight_layout()
plt.savefig("squirrel_diffusion_progress.png")
print("Progress grid saved as squirrel_diffusion_progress.png")
