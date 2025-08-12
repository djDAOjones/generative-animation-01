import os
import re
import csv
import glob
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    DDIMScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    HeunDiscreteScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
)
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# --- CONFIGURABLE ---
DEFAULT_SEEDS = [0, 13, 20100731, 42, 123456]
LETTERS = ['A', 'B', 'C', 'D', 'E']
PALETTE_CSV = os.path.join(os.path.dirname(__file__), "Thread Maps Lookup.csv")
EXPORT_DIR = "exports"
LORA_PATH = "pixel-art-xl.safetensors"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IMG_SIZE = 1024
TARGET_SIZE = (64, 64)
K_CLUSTERS = 16

# --- UTILS ---
def extract_keywords(prompt, max_words=5):
    stopwords = {"the", "a", "an", "with", "of", "and", "in", "on", "for", "to", "by", "at", "as", "is", "from", "this", "that", "no", "not"}
    words = re.sub(r'[^\w\s]', '', prompt.lower()).split()
    keywords = [w for w in words if w not in stopwords]
    return "_".join(keywords[:max_words])

def load_palette(csv_path):
    allowed_colors = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            hex_color = row[2].strip()
            include = row[4].strip().lower()
            if include == "y" and hex_color:
                allowed_colors.append(hex_color)
    palette = [tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) for h in [c.lstrip('#') for c in allowed_colors]]
    return palette

def kmeans_quantize(img, palette, k_clusters=16):
    arr = np.array(img.convert('RGB'))
    flat = arr.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k_clusters, n_init=4, random_state=42)
    labels = kmeans.fit_predict(flat)
    centers = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)
    palette_arr = np.array(palette)
    cluster_to_palette = {}
    for i, center in enumerate(centers):
        dists = np.sum((palette_arr - center) ** 2, axis=1)
        nearest = np.argmin(dists)
        cluster_to_palette[i] = palette_arr[nearest]
    quant = np.array([cluster_to_palette[label] for label in labels]).reshape(arr.shape).astype(np.uint8)
    return Image.fromarray(quant)

def strict_numbering():
    root_dir = os.path.abspath(os.path.dirname(__file__))
    last_file = None
    last_num = 1
    for f in os.listdir(root_dir):
        if f.startswith("LAST_EXPORT_") and f[12:16].isdigit():
            last_file = f
            last_num = int(f[12:16])
            break
    num = last_num + 1
    if last_file:
        os.remove(os.path.join(root_dir, last_file))
    new_last_file = f"LAST_EXPORT_{num:04d}"
    open(os.path.join(root_dir, new_last_file), "w").close()
    return num

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def quantize_to_palette(img, palette):
    arr = np.array(img.convert('RGB'))
    palette_arr = np.array(palette)
    flat = arr.reshape(-1, 3)
    dists = np.sum((flat[:, None, :] - palette_arr[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(dists, axis=1)
    quant = palette_arr[nearest].reshape(arr.shape).astype(np.uint8)
    return Image.fromarray(quant)

def main():
    # --- Interactive input restored ---
    import random
    print("Pixel Art Generator Batch - Interactive Mode\n")
    prompt = input("Enter your prompt: ").strip()
    if not prompt:
        print("Prompt cannot be empty.")
        return
    desc = extract_keywords(prompt)
    print(f"Short description for filenames: {desc}")
    # Number of versions
    try:
        n_versions = int(input("Number of versions (1-10): ").strip())
        if not (1 <= n_versions <= 10):
            print("Number must be between 1 and 10.")
            return
    except Exception:
        print("Invalid number.")
        return
    # Version types
    version_types = ['native', 'quant', 'kmeans']
    print("Version types:")
    for i, vt in enumerate(version_types, 1):
        print(f"  {i}. {vt}")
    print("  4. all")
    vtype_input = input("Which version types? (number or name, default all): ").strip().lower() or '4'
    if vtype_input in ['4', 'all', '']:
        vtype_sel = 'all'
    elif vtype_input in ['1', 'native']:
        vtype_sel = 'native'
    elif vtype_input in ['2', 'quant']:
        vtype_sel = 'quant'
    elif vtype_input in ['3', 'kmeans']:
        vtype_sel = 'kmeans'
    else:
        print("Invalid version type.")
        return
    # Native resolution selection
    native_res_choices = [1024, 512, 256, 128, 64, 32, 16]
    print("Native resolutions:")
    for i, r in enumerate(native_res_choices, 1):
        print(f"  {i}. {r}")
    print(f"  {len(native_res_choices)+1}. all")
    res_input = input("Which native resolution? (number or value, default all): ").strip()
    if res_input in [str(len(native_res_choices)+1), "all", ""]:
        native_resolutions = native_res_choices
    elif res_input.isdigit() and 1 <= int(res_input) <= len(native_res_choices):
        native_resolutions = [native_res_choices[int(res_input)-1]]
    elif res_input in [str(r) for r in native_res_choices]:
        native_resolutions = [int(res_input)]
    else:
        print("Invalid resolution selection.")
        return
    # native_resolutions is now always defined before use    # Scheduler selection
    scheduler_classes = [
        ("DDIM", DDIMScheduler),
        ("Euler", EulerDiscreteScheduler),
        ("PNDM", PNDMScheduler),
        ("LMS", LMSDiscreteScheduler),
        ("Heun", HeunDiscreteScheduler),
        ("DDPM", DDPMScheduler),
        ("DPMSolverMulti", DPMSolverMultistepScheduler),
        ("DPMSolverSingle", DPMSolverSinglestepScheduler),
        ("EulerAncestral", EulerAncestralDiscreteScheduler),
        ("KDPM2", KDPM2DiscreteScheduler),
    ]
    sched_names = [name for name, _ in scheduler_classes]
    print("Available schedulers:")
    for i, name in enumerate(sched_names, 1):
        print(f"  {i}. {name}")
    print(f"  {len(sched_names)+1}. all")
    sched_input = input(f"Which scheduler? (number, name, or 'all', default all): ").strip().lower() or str(len(sched_names)+1)
    if sched_input in [str(len(sched_names)+1), 'all', '']:
        sched_sel = sched_names
    elif sched_input.isdigit() and 1 <= int(sched_input) <= len(sched_names):
        sched_sel = [sched_names[int(sched_input)-1]]
    elif sched_input in sched_names:
        sched_sel = [sched_input]
    else:
        print("Invalid scheduler selection.")
        return
    # --- Prepare seeds and letters ---
    seeds = []
    for i in range(n_versions):
        if i < 3:
            seeds.append(DEFAULT_SEEDS[i])
        else:
            seeds.append(random.randint(0, 2**32 - 1))
    letters = LETTERS[:n_versions]
    # --- Prepare output dir and palette ---
    ensure_dir(EXPORT_DIR)
    palette = load_palette(PALETTE_CSV)
    # --- Strict numbering ---
    num = strict_numbering()
    batch_dir = os.path.join(EXPORT_DIR, f"{num:04d}")
    ensure_dir(batch_dir)
    # --- Schedulers to test ---
    scheduler_classes = [
        ("DDIM", DDIMScheduler),
        ("Euler", EulerDiscreteScheduler),
        ("PNDM", PNDMScheduler),
        ("LMS", LMSDiscreteScheduler),
        ("Heun", HeunDiscreteScheduler),
        ("DDPM", DDPMScheduler),
        ("DPMSolverMulti", DPMSolverMultistepScheduler),
        ("DPMSolverSingle", DPMSolverSinglestepScheduler),
        ("EulerAncestral", EulerAncestralDiscreteScheduler),
        ("KDPM2", KDPM2DiscreteScheduler),
    ]
    # --- For each selected scheduler ---
    for sched_name, sched_class in scheduler_classes:
        if sched_name not in sched_sel:
            continue
        print(f"\n=== Using Scheduler: {sched_name} ===")
        # Load SDXL pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipe.load_lora_weights(LORA_PATH)
        pipe.fuse_lora()
        pipe.scheduler = sched_class.from_config(pipe.scheduler.config)
        pipe.to("mps")
        # --- Generate images for each seed/letter ---
        for i, (seed, letter) in enumerate(zip(seeds, letters)):
            print(f"Generating option {letter} (seed hidden) with {sched_name}...")
            generator = torch.manual_seed(seed)
            fname_base = f"{num:04d}_{letter}_{desc}_{sched_name}"
            # Now use the selected native_resolutions for generation
            for res in native_resolutions:
                print(f"Generating at native resolution {res}x{res} with {sched_name} ...")
                img = pipe(prompt, height=res, width=res, generator=torch.manual_seed(seed)).images[0]
                # Crop to square (should already be, but for safety)
                w, h = img.size
                min_dim = min(w, h)
                img = img.crop(((w-min_dim)//2, (h-min_dim)//2, (w+min_dim)//2, (h+min_dim)//2))
                # Save only selected version types
                if vtype_sel in ('native', 'all'):
                    out_path = os.path.join(batch_dir, f"{fname_base}_{res}x_1_native.png")
                    img_up = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
                    img_up.save(out_path)
                    print(f"Saved {out_path}")
                if vtype_sel in ('quant', 'all'):
                    img_quant = quantize_to_palette(img, palette)
                    img_quant_64 = img_quant.resize((64, 64), resample=Image.NEAREST)
                    img_quant_up = img_quant_64.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
                    out_path_quant = os.path.join(batch_dir, f"{fname_base}_{res}x_2_quant_palette.png")
                    img_quant_up.save(out_path_quant)
                    print(f"Saved {out_path_quant}")
                if vtype_sel in ('kmeans', 'all'):
                    img_kmeans = kmeans_quantize(img, palette, k_clusters=K_CLUSTERS)
                    img_kmeans_64 = img_kmeans.resize((64, 64), resample=Image.NEAREST)
                    img_kmeans_up = img_kmeans_64.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
                    out_path_kmeans = os.path.join(batch_dir, f"{fname_base}_{res}x_3_kmeans_palette.png")
                    img_kmeans_up.save(out_path_kmeans)
                    print(f"Saved {out_path_kmeans}")
        print(f"Updated strict numbering to {num:04d} (dummy file: LAST_EXPORT_{num:04d})")

    # After all generation, run the flexible grid script
    print("\nRunning flexible grid layout script...")
    import subprocess
    try:
        subprocess.run(["python3", "combine_flexible_grid.py", "--export_dir", batch_dir], check=True)
    except Exception as e:
        print(f"Grid script failed: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess cancelled by user (KeyboardInterrupt). Exiting cleanly.")
