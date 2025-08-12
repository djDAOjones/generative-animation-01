"""
Combine images into a grid:
- Rows: version type (native, quant_palette, kmeans_palette)
- Columns: scheduler names (as found in filenames)
- Only uses images with prefix 0017 and native resolution 1024x1024
- White space (32px) between cells
- Each cell is 1024x1024, images centered
- Output: exports/versions_grid_0017.png
"""
import os
import re
from PIL import Image, ImageDraw

EXPORT_DIR = "exports"
START_PREFIX = "0017"
RESOLUTION = 1024
SPACING = 32  # px between cells
CELL_SIZE = 1024

VERSIONS = [
    ("native", "_1_native.png"),
    ("quant_palette", "_2_quant_palette.png"),
    ("kmeans_palette", "_3_kmeans_palette.png"),
]

# Collect files for each version and scheduler
sched_to_files = {v[0]: {} for v in VERSIONS}
all_schedulers = set()
for vtype, vsuffix in VERSIONS:
    pattern = re.compile(rf"{START_PREFIX}_[^_]+_[^_]+_(.+)_{RESOLUTION}x{re.escape(vsuffix)}")
    for fname in os.listdir(EXPORT_DIR):
        if not fname.startswith(START_PREFIX) or not fname.endswith(vsuffix):
            continue
        match = pattern.match(fname)
        if match:
            sched = match.group(1)
            sched_to_files[vtype][sched] = fname
            all_schedulers.add(sched)

schedulers = sorted(all_schedulers)
rows = VERSIONS
cols = schedulers
num_rows = len(rows)
num_cols = len(cols)
grid_w = num_cols * CELL_SIZE + (num_cols + 1) * SPACING
grid_h = num_rows * CELL_SIZE + (num_rows + 1) * SPACING

canvas = Image.new("RGBA", (grid_w, grid_h), (255,255,255,255))
draw = ImageDraw.Draw(canvas)

for row_idx, (vtype, _) in enumerate(rows):
    for col_idx, sched in enumerate(cols):
        fname = sched_to_files[vtype].get(sched)
        x0 = SPACING + col_idx * (CELL_SIZE + SPACING)
        y0 = SPACING + row_idx * (CELL_SIZE + SPACING)
        if fname:
            img = Image.open(os.path.join(EXPORT_DIR, fname)).convert("RGBA")
            img_w, img_h = img.size
            paste_x = x0 + (CELL_SIZE - img_w)//2
            paste_y = y0 + (CELL_SIZE - img_h)//2
            canvas.paste(img, (paste_x, paste_y))
        else:
            draw.rectangle([x0, y0, x0+CELL_SIZE-1, y0+CELL_SIZE-1], outline=(255,0,0,255), width=5)

out_path = os.path.join(EXPORT_DIR, f"versions_grid_{START_PREFIX}.png")
canvas.save(out_path)
print(f"Saved grid: {out_path}")
