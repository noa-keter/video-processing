"""
FinalProject pipeline entry point.
Run from the project root:
    python Code/main.py
All paths are resolved relative to THIS file, so the script works regardless of
the current working directory (no absolute paths are used, per the assignment).
"""
import os
import json
import time
import matting
import tracking
import stabilization as stab
import background_substruction as bgs

# --- Student IDs ------------------------------------------------------------- #
ID1 = "318875770"
ID2 = "322641135"

# --- Project paths (relative to this file) ----------------------------------- #
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
INPUT_DIR = os.path.join(ROOT_DIR, "Inputs")
OUTPUT_DIR = os.path.join(ROOT_DIR, "Outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
INPUT_VIDEO = os.path.join(INPUT_DIR, "INPUT.avi")
BACKGROUND_IMAGE = os.path.join(INPUT_DIR, "background.jpg")
STABILIZED_VIDEO = os.path.join(OUTPUT_DIR, f"stabilize_{ID1}_{ID2}.avi")
EXTRACTED_VIDEO = os.path.join(OUTPUT_DIR, f"extracted_{ID1}_{ID2}.avi")
BINARY_VIDEO = os.path.join(OUTPUT_DIR, f"binary_{ID1}_{ID2}.avi")
MATTED_VIDEO = os.path.join(OUTPUT_DIR, f"matted_{ID1}_{ID2}.avi")
ALPHA_VIDEO = os.path.join(OUTPUT_DIR, f"alpha_{ID1}_{ID2}.avi")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, f"OUTPUT_{ID1}_{ID2}.avi")
TIMING_JSON = os.path.join(OUTPUT_DIR, "timing.json")
TRACKING_JSON = os.path.join(OUTPUT_DIR, "tracking.json")

def main():
    t0 = time.time()
    timing = {}

    print(f"[stabilization] reading {INPUT_VIDEO}")
    transforms, meta = stab.stabilize(INPUT_VIDEO, STABILIZED_VIDEO, fourcc='XVID')
    timing["time_to_stabilize"] = time.time() - t0
    print(f"[stabilization] wrote {STABILIZED_VIDEO}")
    print(f"[stabilization] elapsed: {timing['time_to_stabilize']:.1f}s")

    print(f"[background_substruction] extracting person")
    bgs.run(INPUT_VIDEO, transforms, meta, EXTRACTED_VIDEO, BINARY_VIDEO, fourcc='XVID')
    timing["time_to_binary"] = time.time() - t0
    print(f"[background_substruction] wrote {EXTRACTED_VIDEO} and {BINARY_VIDEO}")
    print(f"[background_substruction] elapsed: {timing['time_to_binary']:.1f}s")

    print(f"[matting] compositing person over {BACKGROUND_IMAGE}")
    matting.run_matting(STABILIZED_VIDEO, BINARY_VIDEO, BACKGROUND_IMAGE, MATTED_VIDEO, ALPHA_VIDEO)
    # matted.avi and alpha.avi are written frame-by-frame in the same pass,
    # so both finish at the same time.
    timing["time_to_alpha"] = time.time() - t0
    timing["time_to_matted"] = timing["time_to_alpha"]
    print(f"[matting] wrote {MATTED_VIDEO} and {ALPHA_VIDEO}")
    print(f"[matting] elapsed: {timing['time_to_matted']:.1f}s")

    print(f"[tracking] tracking person in {MATTED_VIDEO}")
    tracking.run_tracking(MATTED_VIDEO, BINARY_VIDEO, OUTPUT_VIDEO, TRACKING_JSON)
    timing["time_to_output"] = time.time() - t0
    print(f"[tracking] wrote {OUTPUT_VIDEO} and {TRACKING_JSON}")
    print(f"[tracking] elapsed: {timing['time_to_output']:.1f}s")

    with open(TIMING_JSON, "w") as fp:
        json.dump(timing, fp, indent=4)
    print(f"[main] wrote {TIMING_JSON}; total runtime: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
