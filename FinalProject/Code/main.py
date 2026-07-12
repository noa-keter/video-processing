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
import background_substruction as bgs

# --- Student IDs ------------------------------------------------------------- #
ID1 = "ID1"
ID2 = "ID2"

# --- Project paths (relative to this file) ----------------------------------- #
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
INPUT_DIR = os.path.join(ROOT_DIR, "Inputs")
OUTPUT_DIR = os.path.join(ROOT_DIR, "Outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
INPUT_VIDEO = os.path.join(INPUT_DIR, "INPUT.avi")
STABILIZED_VIDEO = os.path.join(OUTPUT_DIR, f"stabilize_{ID1}_{ID2}.avi")
EXTRACTED_VIDEO = os.path.join(OUTPUT_DIR, f"extracted_{ID1}_{ID2}.avi")
BINARY_VIDEO = os.path.join(OUTPUT_DIR, f"binary_{ID1}_{ID2}.avi")

def main():
    t0 = time.time()
    print(f"[stabilization] reading {INPUT_VIDEO}")
    transforms, meta = stab.stabilize(INPUT_VIDEO, STABILIZED_VIDEO, fourcc='XVID')
    n_frames, fps, w, h = meta
    print(f"[stabilization] wrote {STABILIZED_VIDEO}")
    print(f"[stabilization] stage runtime: {time.time() - t0:.1f}s")
    print(f"[background_substruction] extracting person")
    bgs.run(INPUT_VIDEO, transforms, meta, EXTRACTED_VIDEO, BINARY_VIDEO, fourcc='XVID')
    print(f"[background_substruction] wrote {EXTRACTED_VIDEO} and {BINARY_VIDEO}")
    print(f"[background_substruction] stage runtime: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()