# Pixel Art Generator

This project generates pixel art images using Stable Diffusion XL and custom quantization for low-resolution (e.g., 64x64) output. It is optimized for Apple Silicon (M1/M2) GPU (MPS) and now supports automated multi-scheduler, multi-version generation for benchmarking and creative exploration.

## Quick Start

### 1. Prerequisites
- Python 3.10+
- Apple Silicon Mac (M1/M2 recommended for GPU acceleration)
- [Homebrew](https://brew.sh/) (recommended for installing Python)

### 2. Running the Project

Use the provided `run.sh` script to automatically set up and launch the generator:

```bash
bash run.sh
```

This script will:
- Navigate to the project directory
- Create and activate a Python virtual environment (`.venv`) if it doesn't exist
- Upgrade pip and install all required dependencies from `requirements.txt`
- Launch the interactive image generation script (`pixelart_batch_generate_quantize.py`)

### 3. Interactive Usage
- When prompted, enter your image prompt (e.g., `orange cat`).
- Enter the number of versions/options you want (1-10).
- Select the **version type** (native, quant, kmeans, or all) by number (1-4) or by name.
- Select the **native resolution** by number (1-8) or by value. The supported resolutions are:

  1. 1024
  2. 512
  3. 256
  4. 128
  5. 64
  6. 32
  7. 16
  8. all (default)

- Select the **scheduler** by number (1-11) or by name. The supported schedulers are:

  1. DDIM
  2. Euler
  3. PNDM
  4. LMS
  5. Heun
  6. DDPM
  7. DPMSolverMulti
  8. DPMSolverSingle
  9. EulerAncestral
  10. KDPM2
  11. all (default)

- All options can be selected by number or name. Blank input defaults to "all".
- The script will generate images for the selected combinations only, and all menus use numerical prefixes for fast entry.
- For each scheduler and each resolution, **three versions** are produced (unless you select a specific type):
  - Native (model output, upsampled to 1024x1024 with hard edges)
  - Palette quantized (color-constrained to thread palette, downsampled to 64x64 then upsampled to 1024x1024 using nearest neighbor for blocky, pixel-art effect)
  - K-means quantized (palette-constrained using clustering, also downsampled to 64x64 then upsampled to 1024x1024 with nearest neighbor)
- All outputs are saved in the `exports/` directory with filenames indicating scheduler, resolution, processing type, and batch number where relevant.
- All **grid outputs** (including legacy, paged, and combined grids) are saved directly in `exports/` with filenames prepended by `grid_` for easy identification. Grid outputs are uniquely named with batch, scheduler, and/or page information to avoid overwriting.
- You can cancel generation at any time with Ctrl+C (KeyboardInterrupt) for a friendly exit.
- For rapid workflow/dependency testing, use `python3 quick_scheduler_test.py` to generate a test image for each scheduler.
- After generation, a flexible grid of outputs is created automatically. The script arranges images by scheduler (columns), resampling version (rows), native resolution (pages), and seed (pages if needed), with all variables labeled on the grid. 
- Grid outputs are always saved in the main `exports/` directory, never in batch subfolders.
- When run, the grid script will prompt for a 4-digit batch number. If you leave it blank, it will auto-select the most recent batch. Only images from the selected batch are included in the grid.
- You can also run the script manually: `python3 combine_flexible_grid.py`. The output grid(s) will be saved in `exports/` with filenames like `grid_<batch>_<scheduler>_flexgrid_pageN.png`, ensuring uniqueness and clarity.
- The older `combine_kmeans_grid.py` (rows: native resolutions, columns: schedulers) is still available for legacy grid layouts.

## File Structure
- `run.sh` — Standard script to set up the environment and run the generator
- `pixelart_batch_generate_quantize.py` — Main interactive batch script (multi-scheduler, multi-version)
- `quick_scheduler_test.py` — Script to quickly verify all schedulers and dependencies
- `combine_kmeans_grid.py` — Script to generate a grid image of kmeans outputs (rows: resolutions, cols: schedulers)
- `.venv/` — Python virtual environment (auto-created)
- `requirements.txt` — List of pinned Python dependencies
- `Thread Maps Lookup.csv` — Color palette lookup for quantization
- `exports/` — Output directory for generated images and grid exports. All grid outputs are saved here, and intermediate files are automatically cleaned up.

## Grid Output Cleanup
Intermediate files like `grid_flexgrid_combined_page*.png` are automatically cleaned up after each batch or single gridder run, ensuring a tidy output directory.

## Restarting After Reboot or Closing IDE
1. Open a terminal and navigate to the project directory.
2. Run `bash run.sh` to automatically set up and launch the generator.

## Schedulers Supported
The following schedulers are tested and supported for Apple Silicon (M1/M2) GPU workflows:
- DDIM
- Euler
- PNDM
- LMS
- Heun
- DDPM
- DPMSolverMultistep
- DPMSolverSinglestep
- EulerAncestral
- KDPM2

## Output Naming

### Individual Image Outputs
Output files are named as follows:
```
<export_number>_<option_letter>_<prompt_short>_<scheduler>_<resolution>x_<versiontype>.png
```
Where `<versiontype>` is one of:
- `native` (model output)
- `quant_palette` (palette quantized)
- `kmeans_palette` (k-means quantized)

### Grid Outputs
All grid outputs (legacy, paged, combined) are saved in `exports/` with filenames like:
```
grid_<batch>_<prompt>_<scheduler>_flexgrid_pageN.png
grid_legacy_flexgrid_<batch>.png
grid_flexgrid_combined_pageN.png
```
Filenames always begin with `grid_` for easy identification. Batch number, scheduler, and/or page info are included for uniqueness.

### Automatic Cleanup of Intermediate Grid Files
Intermediate files such as `grid_flexgrid_combined_page*.png` and `flexgrid_combined_page*.png` are automatically deleted from `exports/` after each batch or single gridder run. Only the final grid outputs remain for easy access and organization.
## Notes
- The script is configured to use your Mac's GPU (MPS) by default. No extra environment variables are needed.
- Native resolutions now include 32 as an option, and all menus use numerical prefixes for clarity and speed.
- If you want to automate further (aliases, desktop shortcuts, etc.), see the comments in `run.sh` or ask for help.

## Additional Scripts
- `quick_scheduler_test.py`: Run this script to quickly test all scheduler backends and dependencies. Output is saved in `quick_test_outputs/`.
- `combine_kmeans_grid.py`: After generating images, run this script to create a grid of kmeans images (rows: native resolutions, columns: schedulers, each cell is 1024x1024, images centered, white space between cells). Output is saved as `exports/kmeans_grid_<export_number>.png`.

---
**For any issues or questions, please contact the project maintainer or your AI assistant.**
