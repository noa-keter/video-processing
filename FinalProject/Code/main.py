"""
FinalProject pipeline entry point.

Run from the project root:
    python Code/main.py

All paths are resolved relative to THIS file, so the script works regardless of
the current working directory (no absolute paths are used, per the assignment).
"""

import os
import time

import stabilization as stab   

# --- Student IDs ------------------------------------------------------------- #
ID1 = "322641135"
ID2 = "318875770"

# --- Project paths (relative to this file) ----------------------------------- #
CODE_DIR = os.path.dirname(os.path.abspath(__file__))     
ROOT_DIR = os.path.dirname(CODE_DIR)                      
INPUT_DIR = os.path.join(ROOT_DIR, "Inputs")
OUTPUT_DIR = os.path.join(ROOT_DIR, "Outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_VIDEO = os.path.join(INPUT_DIR, "INPUT.avi")

STABILIZED_VIDEO = os.path.join(OUTPUT_DIR, f"stabilize_{ID1}_{ID2}.avi")


def main():
    t0 = time.time()

    print(f"[stabilization] reading {INPUT_VIDEO}")
    transforms, meta = stab.stabilize(INPUT_VIDEO, STABILIZED_VIDEO, fourcc='XVID')
    n_frames, fps, w, h = meta
    print(f"[stabilization] wrote {STABILIZED_VIDEO}")
    print(f"[stabilization] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()