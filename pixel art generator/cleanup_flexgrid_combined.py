import os

EXPORTS_DIR = "exports"
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
