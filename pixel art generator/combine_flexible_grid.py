"""
Flexible Grid Generator for Pixel Art Outputs

- Arranges outputs as a grid with variable hierarchy:
  1. Scheduler version (columns, labeled)
  2. Resampling version (rows, labeled)
  3. Native resolution (pages, labeled)
  4. Seed version (pages, labeled if needed)
- If more than one page is needed for native res or seed, generates multiple files.
- All variables are labeled on the grid.
- Designed to be called from the batch script after generation.

Usage:
    python3 combine_flexible_grid.py [--export_dir EXPORT_DIR] [--out OUT_PREFIX]

"""
import os
import re
import math
from PIL import Image, ImageDraw, ImageFont
import argparse

EXPORT_DIR = "exports"
OUT_PREFIX = "flexgrid"
CELL_SIZE = 1024
SPACING = 32
LABEL_HEIGHT = 80
FONT_SIZE = 48

# Try to load a default font
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
except Exception:
    FONT = ImageFont.load_default()

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=str, default=EXPORT_DIR)
    parser.add_argument('--out', type=str, default=OUT_PREFIX)
    return parser.parse_args()

def prompt_for_batch(records):
    batch_numbers = sorted({r['export_num'] for r in records})
    if not batch_numbers:
        return None
    print(f"Available batch numbers: {', '.join(batch_numbers)}")
    batch = input(f"Which 4-digit batch number? (blank for most recent: {batch_numbers[-1]}): ").strip()
    if batch and batch in batch_numbers:
        return batch
    else:
        return batch_numbers[-1]

def parse_filename(fname):
    # Example: 0017_A_orange_cat_DDIM_1024x_1_native.png
    m = re.match(r"(\d{4})_([A-Z])_([\w_]+)_([\w]+)_([\d]+)x_([123])_([\w]+)\.png", fname)
    if not m:
        return None
    export_num, letter, prompt_short, sched, res, vnum, vtype = m.groups()
    return {
        'fname': fname,
        'sched': sched,
        'vtype': vtype,
        'res': int(res),
        'seed': letter,
        'export_num': export_num
    }

def collect_outputs(export_dir):
    files = [f for f in os.listdir(export_dir) if f.endswith('.png')]
    records = [parse_filename(f) for f in files]
    records = [r for r in records if r]
    return records

def get_unique(records, key):
    return sorted(list({r[key] for r in records}))

def grid_for(records, scheds, vtypes, ress, seeds, export_dir, out_prefix):
    # For each (res, seed) page
    for res in ress:
        for seed in seeds:
            grid_w = len(scheds) * CELL_SIZE + (len(scheds)+1)*SPACING
            grid_h = len(vtypes) * CELL_SIZE + (len(vtypes)+1)*SPACING + LABEL_HEIGHT
            canvas = Image.new("RGBA", (grid_w, grid_h), (255,255,255,255))
            draw = ImageDraw.Draw(canvas)
            # Draw column labels (schedulers)
            for col, sched in enumerate(scheds):
                x = SPACING + col* (CELL_SIZE+SPACING)
                draw.text((x+CELL_SIZE//2, 10), sched, font=FONT, fill=(0,0,0,255), anchor="ma")
            # Draw row labels (vtypes)
            for row, vtype in enumerate(vtypes):
                y = LABEL_HEIGHT + SPACING + row*(CELL_SIZE+SPACING)
                draw.text((10, y+CELL_SIZE//2), vtype, font=FONT, fill=(0,0,0,255), anchor="lm")
            # Place images
            for row, vtype in enumerate(vtypes):
                for col, sched in enumerate(scheds):
                    rec = next((r for r in records if r['sched']==sched and r['vtype']==vtype and r['res']==res and r['seed']==seed), None)
                    x = SPACING + col*(CELL_SIZE+SPACING)
                    y = LABEL_HEIGHT + SPACING + row*(CELL_SIZE+SPACING)
                    if rec:
                        img = Image.open(os.path.join(export_dir, rec['fname'])).convert("RGBA")
                        img_w, img_h = img.size
                        paste_x = x + (CELL_SIZE-img_w)//2
                        paste_y = y + (CELL_SIZE-img_h)//2
                        canvas.paste(img, (paste_x, paste_y))
                    else:
                        draw.rectangle([x, y, x+CELL_SIZE-1, y+CELL_SIZE-1], outline=(255,0,0,255), width=5)
            # Page label
            page_label = f"res={res}, seed={seed}"
            draw.text((grid_w//2, grid_h-LABEL_HEIGHT//2), page_label, font=FONT, fill=(0,0,0,255), anchor="mm")
            # Save
            outname = f"{out_prefix}_res{res}_seed{seed}.png"
            outpath = os.path.join(export_dir, outname)
            canvas.save(outpath)
            print(f"Saved {outpath}")

def main():
    args = parse_args()
    records = collect_outputs(args.export_dir)
    if not records:
        print("No outputs found.")
        return
    batch = prompt_for_batch(records)
    if not batch:
        print("No valid batch numbers found.")
        return
    records = [r for r in records if r['export_num'] == batch]
    print(f"Using batch number: {batch}")
    if not records:
        print("No outputs found for the selected batch.")
        return
    scheds = get_unique(records, 'sched')
    vtypes = get_unique(records, 'vtype')
    ress = get_unique(records, 'res')
    seeds = get_unique(records, 'seed')
    print(f"Schedulers: {scheds}\nResampling: {vtypes}\nResolutions: {ress}\nSeeds: {seeds}")
    grid_for(records, scheds, vtypes, ress, seeds, args.export_dir, args.out)

if __name__ == "__main__":
    main()
