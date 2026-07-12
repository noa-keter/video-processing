"""
Background subtraction on the stabilized video

Two stages:

1. Background model. Learn what each pixel usually looks like over time. Call it
   foreground when the current value disagrees. We use OpenCV's KNN subtractor: 
   it keeps a short per-pixel history and votes background when enough nearby
   samples match in color. Its shadow flag catches pixels that are just a darker
   copy of the background - we drop those so the floor shadow stays out.

2. Color-model refinement. KNN is sure deep inside the person and far outside,
   but fuzzy at the edge. So we build a foreground and a background color histogram
   from the confident regions of THIS frame and let the MAP rule settle the uncertain
   band - higher likelihood wins. The histograms are rebuilt every frame, so they
   follow the person's clothes and the scene rather than relying on fixed colors.

The camera is static after stabilization, so we make two passes. Pass one warms up
KNN over the whole clip and measures the person's height (so the morphology scales
to the person instead of using fixed pixel counts). Pass two builds the masks: keep
the main blob plus any nearby part (a head or leg that split off as the person walks
in), drop distant noise. A final recall-safe step fills any pixel that's foreground
in both neighboring frames.

Outputs:
  extracted_ID1_ID2.avi - person in real color, black elsewhere
  binary_ID1_ID2.avi    - foreground = 1 (stored as 255), background = 0
"""

import cv2
import numpy as np
from scipy import ndimage


def _ellipse(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _odd(value, minimum):
    """Nearest odd integer >= minimum (morphology kernels need odd sizes)."""
    return max(minimum, int(round(value)) | 1)


def _keep_person(mask, max_gap, min_size=150):
    """
    Keep the main blob plus any blob within `max_gap` pixels of it; drop the rest.
    A stray body part (a head or leg that split off as the person walks in) always
    hugs the body, while background false positives sit off on their own - so we test
    proximity, not size. That still catches a small part when the person is far away.
    """
    mask = (mask > 0).astype(np.uint8)
    labels, num_blobs = ndimage.label(mask)
    if num_blobs == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, num_blobs + 1))
    body = 1 + int(np.argmax(sizes))
    near_body = cv2.dilate((labels == body).astype(np.uint8), _ellipse(max_gap)).astype(bool)
    keep = (labels == body)
    for blob in range(1, num_blobs + 1):
        if blob == body:
            continue
        if sizes[blob - 1] >= min_size and (near_body & (labels == blob)).any():
            keep |= (labels == blob)
    return ndimage.binary_fill_holes(keep).astype(np.uint8)


def _clean(raw_mask, close_size, keep_dist, open_size=3):
    """Noisy mask -> the person's region (denoise, bridge gaps, keep body parts)."""
    mask = cv2.morphologyEx((raw_mask > 0).astype(np.uint8), cv2.MORPH_OPEN, _ellipse(open_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(close_size))
    return _keep_person(mask, keep_dist)


def _color_map_refine(frame, mask, erode_fg, dilate_bg, keep_dist, bins=32):
    """
    Section-8 color-model + MAP cleanup of the mask boundary.
    Sure foreground = the mask's interior (eroded); sure background = well outside it
    (dilated, then inverted); the band between is uncertain. Build a foreground and a
    background color histogram from the sure regions of this frame, then hand each
    uncertain pixel to whichever model likes it more (the MAP rule).
    """
    sure_fg = cv2.erode(mask, _ellipse(erode_fg)).astype(bool)
    sure_bg = ~cv2.dilate(mask, _ellipse(dilate_bg)).astype(bool)
    unknown = (~sure_fg) & (~sure_bg)
    if sure_fg.sum() < 500 or sure_bg.sum() < 500:
        return mask

    bin_width = 256 // bins
    binned = (frame // bin_width).astype(np.int32)
    bin_idx = (binned[..., 0] * bins + binned[..., 1]) * bins + binned[..., 2]   # color -> histogram bin
    hist_fg = np.bincount(bin_idx[sure_fg], minlength=bins ** 3).astype(np.float32)
    hist_bg = np.bincount(bin_idx[sure_bg], minlength=bins ** 3).astype(np.float32)
    hist_fg /= hist_fg.sum() + 1e-9
    hist_bg /= hist_bg.sum() + 1e-9

    refined = sure_fg | (unknown & (hist_fg[bin_idx] > hist_bg[bin_idx]))   # MAP: foreground if it wins
    refined = cv2.morphologyEx(refined.astype(np.uint8), cv2.MORPH_OPEN, _ellipse(3))
    return _keep_person(refined, keep_dist)


def _temporal_fill(masks, keep_dist):
    """
    Recall-safe temporal smoothing: add back any pixel that's foreground in BOTH
    neighbors (recovers a one-frame drop-out) and never remove anything, so
    fast-moving parts survive.
    """
    filled = []
    num_frames = len(masks)
    for i in range(num_frames):
        mask = masks[i].copy()
        if 0 < i < num_frames - 1:
            mask = (mask | (masks[i - 1] & masks[i + 1])).astype(np.uint8)
        filled.append(_keep_person(mask, keep_dist))
    return filled


def run(stabilized_path, transforms, meta, extracted_path, binary_path,
        dist2=300.0, fourcc='XVID'):
    """
    Build the KNN background model, classify + color-refine every stabilized frame,
    steady the masks in time, and write the extracted (color) and binary videos.
    Reads the already-written stabilized video once; the transforms are only used
    to reproduce each frame's valid-pixel mask. Returns the list of foreground masks.
    """
    num_frames, fps, width, height = meta

    cap = cv2.VideoCapture(stabilized_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    full_mask = np.full((height, width), 255, np.uint8)
    valids = [cv2.warpAffine(full_mask, transform, (width, height), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
              for transform in transforms]
    # dist2Threshold is the color-distance cutoff for "same as background". It only
    # roughs out the mask - the color refinement redraws the boundary each frame - so
    # the exact value doesn't matter much. This is the one knob we tune.
    bg_model = cv2.createBackgroundSubtractorKNN(history=num_frames, dist2Threshold=dist2,
                                                 detectShadows=True)

    # pass 1: warm up the model, then (once warm, over the second half) measure the
    # person's HEIGHT so every size below scales to the person instead of fixed pixels.
    heights = []
    for i, (frame, valid) in enumerate(zip(frames, valids)):
        fg_mask = bg_model.apply(frame)
        if i >= num_frames // 2:
            mask = cv2.morphologyEx(((fg_mask == 255) & valid).astype(np.uint8),
                                    cv2.MORPH_OPEN, _ellipse(3))
            labels, num_blobs = ndimage.label(mask)
            if num_blobs > 0:
                body = 1 + int(np.argmax(ndimage.sum(mask, labels, range(1, num_blobs + 1))))
                rows = np.where(labels == body)[0]
                heights.append(rows.max() - rows.min())
    person_height = float(np.median(heights)) if heights else 671.0
    close_size = _odd(person_height * 0.016, 5)     # bridge small gaps
    erode_fg = _odd(person_height * 0.030, 7)       # shrink to sure foreground
    dilate_bg = _odd(person_height * 0.045, 9)      # grow the sure-background band
    keep_dist = _odd(person_height * 0.080, 9)      # how close a detached part must be to count as body
    connect_size = _odd(person_height * 0.022, 9)   # close break-gaps (neck/knee) into one region

    # pass 2: classify, clean, and color-refine each frame
    masks = []
    for frame, valid in zip(frames, valids):
        fg_mask = bg_model.apply(frame)            # 255 = foreground, 127 = shadow, 0 = background
        mask = _clean(((fg_mask == 255) & valid).astype(np.uint8), close_size, keep_dist)
        mask = _color_map_refine(frame, mask, erode_fg, dilate_bg, keep_dist)
        # weld the kept parts (a separately-detected head/leg) into one solid region so
        # the extracted person has no gaps; the kernel is small enough to leave the
        # natural gap between spread legs open.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(connect_size))
        masks.append(ndimage.binary_fill_holes(mask).astype(np.uint8))

    masks = _temporal_fill(masks, keep_dist)

    # pass 3: write out, pairing each stabilized frame with its foreground mask
    ex_writer = cv2.VideoWriter(extracted_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    bin_writer = cv2.VideoWriter(binary_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    for i, frame in enumerate(frames):
        mask = masks[i]
        extracted = frame.copy()
        extracted[mask == 0] = 0                   # foreground keeps its real colors, rest goes black
        ex_writer.write(extracted)
        # binary as 3-channel 0/255 so the color codec survives it; matting thresholds it back
        bin_writer.write(cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR))
    ex_writer.release()
    bin_writer.release()
    return masks
