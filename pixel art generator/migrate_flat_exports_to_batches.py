import os
import shutil
import re

EXPORTS_DIR = "exports"
BATCH_REGEX = re.compile(r'^(\d{4})_')

# List all files in exports dir
files = [f for f in os.listdir(EXPORTS_DIR) if os.path.isfile(os.path.join(EXPORTS_DIR, f))]

moved = 0
for fname in files:
    match = BATCH_REGEX.match(fname)
    if match:
        batch = match.group(1)
        batch_dir = os.path.join(EXPORTS_DIR, batch)
        os.makedirs(batch_dir, exist_ok=True)
        src = os.path.join(EXPORTS_DIR, fname)
        dst = os.path.join(batch_dir, fname)
        shutil.move(src, dst)
        moved += 1
        print(f"Moved {fname} -> {batch}/")
    else:
        print(f"Skipped {fname} (no batch number found)")

print(f"Migration complete. {moved} files moved into batch folders.")
