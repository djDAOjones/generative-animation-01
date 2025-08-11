import csv
from PIL import Image
import numpy as np
import os
from sklearn.cluster import KMeans

# --- CONFIG ---
import glob

def find_latest_input():
    files = glob.glob("exports/*_1024x.png")
    if not files:
        raise FileNotFoundError("No *_1024x.png files found in exports/")
    # Sort by file number in prefix
    files.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
    return files[-1]

INPUT_IMAGE = find_latest_input()
PALETTE_CSV = "Thread Maps Lookup.csv"
OUTPUT_IMAGE = "exports/orange_cat_icon_bold_kmeans_64x.png"
TARGET_SIZE = (64, 64)
K_CLUSTERS = 16  # Number of color clusters, can be tuned

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

# --- K-MEANS QUANTIZATION ---
def kmeans_quantize(img, palette, k_clusters=16):
    arr = np.array(img.convert('RGB'))
    flat = arr.reshape(-1, 3)
    # K-means clustering
    kmeans = KMeans(n_clusters=k_clusters, n_init=4, random_state=42)
    labels = kmeans.fit_predict(flat)
    centers = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)
    # Map each cluster center to nearest palette color
    palette_arr = np.array(palette)
    cluster_to_palette = {}
    for i, center in enumerate(centers):
        dists = np.sum((palette_arr - center) ** 2, axis=1)
        nearest = np.argmin(dists)
        cluster_to_palette[i] = palette_arr[nearest]
    # Assign each pixel to mapped palette color
    quant = np.array([cluster_to_palette[label] for label in labels]).reshape(arr.shape).astype(np.uint8)
    return Image.fromarray(quant)

# --- MAIN WORKFLOW ---
def main():
    # Load and crop image
    img = Image.open(INPUT_IMAGE)
    # Downsample to 64x64 (nearest neighbor for pixel art)
    img = img.resize(TARGET_SIZE, resample=Image.NEAREST)
    # Load palette
    palette = load_palette(PALETTE_CSV)
    # K-means quantize
    img_quant = kmeans_quantize(img, palette, k_clusters=K_CLUSTERS)
    # Upscale to 1024x1024 for viewing
    img_quant_up = img_quant.resize((1024, 1024), resample=Image.NEAREST)
    img_quant_up.save(OUTPUT_IMAGE)
    print(f"Saved K-means quantized 64x64 (upscaled) image to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
