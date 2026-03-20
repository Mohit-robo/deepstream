import os
import glob

search_paths = [
    '/opt/nvidia/deepstream/deepstream/lib/',
    '/usr/lib/aarch64-linux-gnu/',
    '/usr/lib/aarch64-linux-gnu/tegra/'
]

for p in search_paths:
    if os.path.exists(p):
        print(f"Checking {p}...")
        results = glob.glob(os.path.join(p, "*nvdcf*"))
        for r in results:
            print(f"FOUND: {r}")
    else:
        print(f"Path not found: {p}")
