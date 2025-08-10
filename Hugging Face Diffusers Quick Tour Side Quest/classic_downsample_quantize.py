import csv
from PIL import Image
import numpy as np
import os

# --- CONFIG ---
INPUT_IMAGE = "exports/orange_cat_icon_bold_1024.png"  # Change as needed
PALETTE_CSV = "Thread Maps Lookup.csv"
OUTPUT_IMAGE = "exports/orange_cat_icon_bold_64_quant.png"
TARGET_SIZE = (64, 64)

# --- LOAD ALLOWED COLORS FROM CSV ---
def load_palette(csv_path):
    allowed_colors = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            hex_color = row[2].strip()
            include = row[4].strip().lower()
            if include == "y" and hex_color:
                allowed_colors.append(hex_color)
    # Convert hex to RGB tuples
    palette = [tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) for h in [c.lstrip('#') for c in allowed_colors]]
    return palette

# --- CROP TO SQUARE ---
def crop_to_square(img):
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    return img.crop((left, top, right, bottom))

# --- QUANTIZE TO PALETTE ---
def quantize_to_palette(img, palette):
    arr = np.array(img.convert('RGB'))
    palette_arr = np.array(palette)
    # Compute distance to each palette color
    flat = arr.reshape(-1, 3)
    dists = np.sum((flat[:, None, :] - palette_arr[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(dists, axis=1)
    quant = palette_arr[nearest].reshape(arr.shape).astype(np.uint8)
    return Image.fromarray(quant)

# --- MAIN WORKFLOW ---
def main():
    # Load and crop image
    img = Image.open(INPUT_IMAGE)
    img = crop_to_square(img)
    # Downsample to 64x64 (nearest neighbor for pixel art)
    img = img.resize(TARGET_SIZE, resample=Image.NEAREST)
    # Load palette
    palette = load_palette(PALETTE_CSV)
    # Quantize
    img_quant = quantize_to_palette(img, palette)
    # Save
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    img_quant.save(OUTPUT_IMAGE)
    print(f"Saved quantized 64x64 image to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
