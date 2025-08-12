#!/bin/bash
set -x
# Standard run script for Pixel Art Generator project

# Navigate to project directory
cd "$(dirname "$0")"

# Create and activate venv if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Run the main script interactively
python3 pixelart_batch_generate_quantize.py
