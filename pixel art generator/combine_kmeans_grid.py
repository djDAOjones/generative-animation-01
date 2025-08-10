"""
Combine kmeans-processed images into a grid:
- Only use exports starting with 0017 and ending with _3_kmeans_palette.png
- Rows: native resolutions (1024, 512, 256, 128, 64)
- Columns: scheduler names (as found in filenames)
- White space (32px) between cells
- Output image is native res per cell, added together (large!)
"""
import os
from PIL import Image, ImageDraw

EXPORT_DIR = "exports"
START_PREFIX = "0017"
KMEANS_SUFFIX = "_3_kmeans_palette.png"
NATIVE_RESOLUTIONS = [1024, 512, 256, 128, 64]
SPACING = 32  # px between cells

# Find all matching files
files = [f for f in os.listdir(EXPORT_DIR) if f.startswith(START_PREFIX) and f.endswith(KMEANS_SUFFIX)]

import re
# Parse scheduler names and resolutions using regex
sched_to_files = {}
for f in files:
    # Example: 0017_A_orange_cat_DPMSolverMulti_1024x_3_kmeans_palette.png
    match = re.match(r"0017_[^_]+_[^_]+_(.+)_(\d+)x_3_kmeans_palette\.png", f)
    if not match:
        print(f"[WARN] Could not parse: {f}")
        continue
    sched = match.group(1)
    res = int(match.group(2))
    if sched not in sched_to_files:
        sched_to_files[sched] = {}
    sched_to_files[sched][res] = f

# Sort scheduler columns by name
schedulers = sorted(sched_to_files.keys())
# Sort resolutions descending (rows)
resolutions = sorted(NATIVE_RESOLUTIONS, reverse=True)

# Compute grid size and cell size
cell_size = max(resolutions)  # 1024
num_rows = len(resolutions)
num_cols = len(schedulers)
grid_w = num_cols * cell_size + (num_cols + 1) * SPACING
grid_h = num_rows * cell_size + (num_rows + 1) * SPACING

# Create blank canvas
canvas = Image.new("RGBA", (grid_w, grid_h), (255,255,255,255))
draw = ImageDraw.Draw(canvas)

# Place images, centering each native res in its cell
for row, res in enumerate(resolutions):
    for col, sched in enumerate(schedulers):
        fname = sched_to_files.get(sched, {}).get(res)
        x0 = SPACING + col * (cell_size + SPACING)
        y0 = SPACING + row * (cell_size + SPACING)
        if fname:
            img = Image.open(os.path.join(EXPORT_DIR, fname)).convert("RGBA")
            img_w, img_h = img.size
            # Center image in cell
            paste_x = x0 + (cell_size - img_w)//2
            paste_y = y0 + (cell_size - img_h)//2
            canvas.paste(img, (paste_x, paste_y))
        else:
            # Optionally, draw a red rectangle for missing images
            draw.rectangle([x0, y0, x0+cell_size-1, y0+cell_size-1], outline=(255,0,0,255), width=5)

out_path = os.path.join(EXPORT_DIR, f"kmeans_grid_{START_PREFIX}.png")
canvas.save(out_path)
print(f"Saved grid: {out_path}")
