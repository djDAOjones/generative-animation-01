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
TILE_MARGIN = 20  # px around each image
SPACING = TILE_MARGIN * 2  # 40px between tiles
METADATA_HEIGHT = 3 * 18  # 3 lines, 18px each (font size 12 + margin)
LABEL_HEIGHT = 40  # for grid title at top
FONT_SIZE = 28

import sys
# Try to use DejaVuSans.ttf from script dir or fonts/ subdir, fallback to any DejaVuSans*.ttf in fonts/
def find_font():
    script_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(script_dir, "DejaVuSans.ttf"),
        os.path.join(script_dir, "fonts", "DejaVuSans.ttf")
    ]
    # Add any DejaVuSans*.ttf in fonts/
    fonts_dir = os.path.join(script_dir, "fonts")
    if os.path.isdir(fonts_dir):
        for fname in os.listdir(fonts_dir):
            if fname.lower().startswith("dejavusans") and fname.lower().endswith(".ttf"):
                candidates.append(os.path.join(fonts_dir, fname))
    for path in candidates:
        if os.path.exists(path):
            return path
    print("ERROR: No DejaVuSans .ttf font found. Looked for:")
    for path in candidates:
        print(f" - {path}")
    print("Please download DejaVuSans.ttf and place it in the script or fonts/ directory.")
    sys.exit(1)

FONT_PATH = find_font()
FONT = ImageFont.truetype(FONT_PATH, FONT_SIZE)

METADATA_HEIGHT = 3 * FONT_SIZE  # 3 lines, exactly font size spacing

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

def grid_for(records, scheds, vtypes, ress, seeds, export_dir, out_prefix, batch=None, prompt=None):
    # Determine grid axes and paging
    # For this implementation, columns = scheds, rows = vtypes, page for each (res, seed)
    for res in ress:
        for seed in seeds:
            ncols = len(scheds)
            nrows = len(vtypes)
            grid_w = ncols * (CELL_SIZE + 2*TILE_MARGIN) + (ncols+1)*SPACING
            grid_h = LABEL_HEIGHT + nrows * (CELL_SIZE + 2*TILE_MARGIN + METADATA_HEIGHT) + (nrows+1)*SPACING
            canvas = Image.new("RGBA", (grid_w, grid_h), (255,255,255,255))
            draw = ImageDraw.Draw(canvas)

            # No grid title (no batch or prompt)

            # Draw images and metadata
            for row, vtype in enumerate(vtypes):
                for col, sched in enumerate(scheds):
                    rec = next((r for r in records if r['sched']==sched and r['vtype']==vtype and r['res']==res and r['seed']==seed), None)
                    x = SPACING + col * (CELL_SIZE + 2*TILE_MARGIN + SPACING)
                    y = LABEL_HEIGHT + SPACING + row * (CELL_SIZE + 2*TILE_MARGIN + METADATA_HEIGHT + SPACING)
                    # Draw tile background (optional)
                    # draw.rectangle([x, y, x+CELL_SIZE+2*TILE_MARGIN, y+CELL_SIZE+2*TILE_MARGIN+METADATA_HEIGHT], fill=(245,245,245,255))
                    if rec:
                        img = Image.open(os.path.join(export_dir, rec['fname'])).convert("RGBA")
                        img_w, img_h = img.size
                        paste_x = x + TILE_MARGIN + (CELL_SIZE-img_w)//2
                        paste_y = y + TILE_MARGIN
                        canvas.paste(img, (paste_x, paste_y))
                        # Metadata lines
                        meta1 = f"{rec['sched']}, {rec['res']}"
                        meta2 = f"Seed: {rec['seed']}"
                        meta3 = f"{rec['vtype']}"
                        meta_x = x + TILE_MARGIN
                        meta_y = y + TILE_MARGIN + CELL_SIZE + 4
                        draw.text((meta_x, meta_y), meta1, font=FONT, fill=(0,0,0,255), anchor="la")
                        draw.text((meta_x, meta_y+FONT_SIZE), meta2, font=FONT, fill=(0,0,0,255), anchor="la")
                        draw.text((meta_x, meta_y+2*FONT_SIZE), meta3, font=FONT, fill=(0,0,0,255), anchor="la")
                    else:
                        draw.rectangle([x+TILE_MARGIN, y+TILE_MARGIN, x+TILE_MARGIN+CELL_SIZE-1, y+TILE_MARGIN+CELL_SIZE-1], outline=(255,0,0,255), width=5)
                        # Optionally, draw missing metadata
            # Save
            outname = f"{out_prefix}_res{res}_seed{seed}.png"
            outpath = os.path.join(export_dir, outname)
            canvas.save(outpath)
            print(f"Saved {outpath}")

def combined_grid(records, scheds, vtypes, ress, seeds, export_dir, out_prefix, batch=None, prompt=None):
    # Settings for title
    TITLE_FONT_SIZE = 48
    TITLE_MARGIN = 40
    title_font = ImageFont.truetype(FONT_PATH, TITLE_FONT_SIZE)
    # We'll page over (res, seed) pairs if >1 of either
    pages = []
    for res in ress:
        for seed in seeds:
            pages.append((res, seed))
    # Try to fit all pages in one grid if possible
    if len(pages) == 1:
        page_groups = [pages]
    else:
        # For each page, make a grid of scheds x vtypes
        # If too many pages, group them (e.g., 4 pages per combined grid)
        group_size = 4  # show up to 4 (res,seed) combos per combined grid
        page_groups = [pages[i:i+group_size] for i in range(0, len(pages), group_size)]
    for idx, group in enumerate(page_groups):
        ncols = len(scheds) * len(group)
        nrows = len(vtypes)
        grid_w = ncols * (CELL_SIZE + 2*TILE_MARGIN) + (ncols+1)*SPACING + 2*TITLE_MARGIN
        grid_h = TITLE_MARGIN + TITLE_FONT_SIZE + TITLE_MARGIN + nrows * (CELL_SIZE + 2*TILE_MARGIN + METADATA_HEIGHT) + (nrows+1)*SPACING + TITLE_MARGIN
        canvas = Image.new("RGBA", (grid_w, grid_h), (255,255,255,255))
        draw = ImageDraw.Draw(canvas)
        # Draw page title (batch + prompt) at top center with 40px margin all around
        title = f"Batch: {batch or ''} | Prompt: {prompt or ''}"
        title_y = TITLE_MARGIN + TITLE_FONT_SIZE//2
        draw.text((grid_w//2, title_y), title, font=title_font, fill=(0,0,0,255), anchor="mm")
        # Draw images and metadata
        for g, (res, seed) in enumerate(group):
            for row, vtype in enumerate(vtypes):
                for col, sched in enumerate(scheds):
                    rec = next((r for r in records if r['sched']==sched and r['vtype']==vtype and r['res']==res and r['seed']==seed), None)
                    x = TITLE_MARGIN + SPACING + (g*len(scheds)+col) * (CELL_SIZE + 2*TILE_MARGIN + SPACING)
                    y = TITLE_MARGIN + TITLE_FONT_SIZE + TITLE_MARGIN + SPACING + row * (CELL_SIZE + 2*TILE_MARGIN + METADATA_HEIGHT + SPACING)
                    if rec:
                        img = Image.open(os.path.join(export_dir, rec['fname'])).convert("RGBA")
                        img_w, img_h = img.size
                        paste_x = x + TILE_MARGIN + (CELL_SIZE-img_w)//2
                        paste_y = y + TILE_MARGIN
                        canvas.paste(img, (paste_x, paste_y))
                        # Metadata lines
                        meta1 = f"{rec['sched']}, {rec['res']}"
                        meta2 = f"Seed: {rec['seed']}"
                        meta3 = f"{rec['vtype']}"
                        meta_x = x + TILE_MARGIN
                        meta_y = y + TILE_MARGIN + CELL_SIZE
                        draw.text((meta_x, meta_y), meta1, font=FONT, fill=(0,0,0,255), anchor="la")
                        draw.text((meta_x, meta_y+FONT_SIZE), meta2, font=FONT, fill=(0,0,0,255), anchor="la")
                        draw.text((meta_x, meta_y+2*FONT_SIZE), meta3, font=FONT, fill=(0,0,0,255), anchor="la")
                    else:
                        draw.rectangle([x+TILE_MARGIN, y+TILE_MARGIN, x+TILE_MARGIN+CELL_SIZE-1, y+TILE_MARGIN+CELL_SIZE-1], outline=(255,0,0,255), width=5)
        outname = f"{out_prefix}_combined_page{idx+1}.png"
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
    grid_for(records, scheds, vtypes, ress, seeds, args.export_dir, args.out, batch=batch, prompt=None)
    # Extract prompt from a representative record if present
    prompt = None
    for r in records:
        if 'prompt_short' in r:
            prompt = r['prompt_short']
            break
        elif 'prompt' in r:
            prompt = r['prompt']
            break
    if not prompt and records:
        # fallback: try to parse from filename
        fname = records[0]['fname']
        parts = fname.split('_')
        if len(parts) > 2:
            prompt = parts[2]
    # Generate combined grid for all images in batch
    combined_grid(records, scheds, vtypes, ress, seeds, args.export_dir, args.out, batch=batch, prompt=prompt)

if __name__ == "__main__":
    main()
