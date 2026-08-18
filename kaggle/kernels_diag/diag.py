"""Diagnostic: dump /kaggle/input contents to see what actually mounts."""

import os

print("=== /kaggle/input ===")
try:
    print(os.listdir("/kaggle/input"))
except Exception as e:
    print("ERR", e)

for root, dirs, files in os.walk("/kaggle/input"):
    print(root, "->", dirs[:8], files[:10])

print("\n=== known candidates ===")
for p in ("/kaggle/input/ebh-jepa-crafter-env",
          "/kaggle/input/datasets/sehajrsingh/ebh-jepa-crafter-env"):
    print(p, "->", os.path.isdir(p))
