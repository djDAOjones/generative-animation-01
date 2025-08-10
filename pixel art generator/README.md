# Pixel Art Generator

This project generates pixel art images using Stable Diffusion XL and custom quantization for low-resolution (e.g., 64x64) output. It is optimized for Apple Silicon (M1/M2) GPU (MPS) and uses the DDIM scheduler for best results at low resolutions.

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
- Enter the number of versions/options you want (1-5).
- Generated images and outputs will be saved in the `exports/` directory.

## File Structure
- `run.sh` — Standard script to set up the environment and run the generator
- `pixelart_batch_generate_quantize.py` — Main interactive script
- `.venv/` — Python virtual environment (auto-created)
- `requirements.txt` — List of pinned Python dependencies
- `Thread Maps Lookup.csv` — Color palette lookup for quantization
- `exports/` — Output directory for generated images

## Restarting After Reboot or Closing IDE
1. Open a terminal and navigate to the project directory.
2. Run `bash run.sh` to automatically set up and launch the generator.

## Notes
- The script is configured to use your Mac's GPU (MPS) by default. No extra environment variables are needed.
- If you want to automate further (aliases, desktop shortcuts, etc.), see the comments in `run.sh` or ask for help.

---

**For any issues or questions, please contact the project maintainer or your AI assistant.**
