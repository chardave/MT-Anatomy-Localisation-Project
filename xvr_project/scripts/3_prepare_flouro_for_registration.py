from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"
OUTPUT = ROOT / "outputs" / "registration"
OUTPUT.mkdir(parents=True, exist_ok=True)

src = INPUT / "fluoroscopy_AP.png"
dst = OUTPUT / "fluoroscopy_AP_prepared.png"

img = Image.open(src).convert("L")
arr = np.array(img)

# ---- 1. crop away the outer border area ----
# adjust if needed, but this is a good first pass
crop = arr[20:500, 10:485]

# ---- 2. blank out text / annotations ----
# top-left text
crop[40:110, 0:120] = 255

# top-right text
crop[0:55, 360:475] = 255

# bottom-left text
crop[410:480, 0:70] = 255

# bottom edge line / border
crop[465:480, :] = 255

# ---- 3. optionally suppress lower anatomy not covered by CT ----
# keep mainly skull + orbits + upper midface
# comment this out if you want to keep more anatomy
crop = crop[0:390, :]

Image.fromarray(crop).save(dst)

print("Saved:", dst)
print("Prepared image shape:", crop.shape)