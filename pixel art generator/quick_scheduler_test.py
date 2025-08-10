"""
Quick Scheduler Test Script
- Loads each supported scheduler
- Generates a single 128x128 image for a fixed prompt and seed
- Prints success/failure for each scheduler
- Use this to quickly test workflow, dependencies, and device setup
"""
import torch
from diffusers import StableDiffusionXLPipeline
import os

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_PATH = "./loras/pixel.safetensors"  # Adjust if needed
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SCHEDULERS = [
    ("DDIM", "diffusers.schedulers.DDIMScheduler"),
    ("Euler", "diffusers.schedulers.EulerDiscreteScheduler"),
    ("PNDM", "diffusers.schedulers.PNDMScheduler"),
    ("LMS", "diffusers.schedulers.LMSDiscreteScheduler"),
    ("Heun", "diffusers.schedulers.HeunDiscreteScheduler"),
    ("DDPM", "diffusers.schedulers.DDPMScheduler"),
    ("DPMSolverMultistep", "diffusers.schedulers.DPMSolverMultistepScheduler"),
    ("DPMSolverSinglestep", "diffusers.schedulers.DPMSolverSinglestepScheduler"),
    ("EulerAncestral", "diffusers.schedulers.EulerAncestralDiscreteScheduler"),
    ("KDPM2", "diffusers.schedulers.KDPM2DiscreteScheduler"),
]

prompt = "orange cat"
seed = 42
height = width = 128
output_dir = "quick_test_outputs"
os.makedirs(output_dir, exist_ok=True)

def get_scheduler_class(sched_path):
    module, cls = sched_path.rsplit(".", 1)
    return getattr(__import__(module, fromlist=[cls]), cls)

for sched_name, sched_path in SCHEDULERS:
    try:
        print(f"\nTesting scheduler: {sched_name} ...", flush=True)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        # Optionally load LoRA
        if os.path.exists(LORA_PATH):
            pipe.load_lora_weights(LORA_PATH)
            pipe.fuse_lora()
        sched_class = get_scheduler_class(sched_path)
        pipe.scheduler = sched_class.from_config(pipe.scheduler.config)
        pipe.to(DEVICE)
        generator = torch.manual_seed(seed)
        img = pipe(prompt, height=height, width=width, generator=generator).images[0]
        out_path = os.path.join(output_dir, f"test_{sched_name}.png")
        img.save(out_path)
        print(f"  [SUCCESS] Saved: {out_path}")
    except Exception as e:
        print(f"  [FAIL] {sched_name}: {e}")
