# This script will process all batch folders in exports/ using combine_flexible_grid.py.
# The final grid exports (flexgrid...) will be saved in the main exports/ folder.
# Any intermediate files (e.g. with captions) will remain in the batch folders.

import os
import subprocess

EXPORTS_DIR = "exports"

batch_folders = [f for f in os.listdir(EXPORTS_DIR) if os.path.isdir(os.path.join(EXPORTS_DIR, f)) and f.isdigit() and len(f) == 4]
batch_folders = sorted(batch_folders)

if not batch_folders:
    print("No batch folders found in exports/.")
    exit(1)

for batch in batch_folders:
    batch_dir = os.path.join(EXPORTS_DIR, batch)
    print(f"Processing batch {batch}...")
    result = subprocess.run([
        "python3", "combine_flexible_grid.py", "--export_dir", batch_dir
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error processing batch {batch}: {result.stderr}")
    else:
        print(f"Batch {batch} completed.\n")

print("All batches processed.")

# Cleanup temp/intermediate files (flexgrid_combined_page*.png) from exports/ and subfolders
def cleanup_flexgrid_combined():
    removed = 0
    for root, dirs, files in os.walk(EXPORTS_DIR):
        for fname in files:
            if fname.startswith("flexgrid_combined_page") and fname.endswith(".png"):
                fpath = os.path.join(root, fname)
                try:
                    os.remove(fpath)
                    print(f"Deleted {fpath}")
                    removed += 1
                except Exception as e:
                    print(f"Failed to delete {fpath}: {e}")
    print(f"Cleanup complete. {removed} files deleted.")

cleanup_flexgrid_combined()
