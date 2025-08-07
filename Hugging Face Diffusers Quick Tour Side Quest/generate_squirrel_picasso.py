from diffusers import DiffusionPipeline, UniPCMultistepScheduler
import torch

# Load the Stable Diffusion pipeline
pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)

# Move to the best available device (MPS for Apple Silicon, CUDA, then CPU)
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

# Swap the scheduler to UniPCMultistepScheduler
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

# Generate an image from a text prompt using the new scheduler
prompt = "An image of a squirrel in Picasso style"
image = pipeline(prompt).images[0]

# Save the image
image.save("image_of_squirrel_unipc.png")
print("Image saved as image_of_squirrel_unipc.png")
