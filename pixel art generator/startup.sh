#!/bin/bash
# Startup script for Pixel Art Generator project

# Navigate to project directory
cd "$(dirname "$0")"

# Create and activate venv if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install requirements if needed
pip install --upgrade pip
pip install -r requirements.txt

# Run the main script
python3 pixelart_batch_generate_quantize.py
