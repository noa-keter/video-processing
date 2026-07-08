"""
Background subtraction on the stabilized video.

This is the per-pixel background modelling from Section 8 of the notes, in two
stages:

1. Background model. Each pixel is described by what it usually looks like over
   time; a pixel is foreground when the current value disagrees with that model.
   We use OpenCV's KNN background subtractor (allowed for this project) - a
   non-parametric version of the same idea: it keeps a short history of recent
   samples per pixel and calls the pixel background when enough of its nearest
   samples are close in colour. Its shadow detector flags pixels that are only a
   darker copy of the background, which we drop so the floor shadow stays out.

2. Colour-model refinement (also Section 8: a likelihood model with a MAP
   decision). The KNN mask is confident deep inside the person and far outside but
   rough at the boundary, so we build a foreground and a background colour model
   from the confident regions of the CURRENT frame and label the uncertain band
   between them by the MAP rule from the notes - whichever model gives the higher
   likelihood wins. The colour models are histograms rebuilt every frame, so they
   adapt to the person's clothes and the scene instead of using fixed colours.

The camera is static after stabilization, so we run two passes: the first warms up
the KNN model over the whole clip (and measures the person's height so the
morphology can scale to the person's size instead of using fixed pixel counts),
the second produces the masks. Each mask is cleaned by keeping the main foreground
region plus any nearby parts (a head or leg detected as a separate blob while the
person is entering) and dropping distant noise. A final recall-safe temporal step
fills any pixel that is foreground in both neighbouring frames.

Outputs:
  extracted_ID1_ID2.avi - foreground (person) in original colour, black elsewhere
  binary_ID1_ID2.avi    - foreground = 1 (stored as 255), background = 0
"""

import cv2
import numpy as np
from scipy import ndimage

import stabilization as stab


def _ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _odd(x, floor):
    """Nearest odd integer >= floor (morphology kernels want odd sizes)."""
    return max(floor, int(round(x)) | 1)


def _keep_person(m, dist, min_abs=150):
    """
    Keep the main foreground blob plus any blob lying within `dist` pixels of it,
    and drop the rest. Body parts (a head or a leg detected as a separate blob when
    the person is entering) always sit right next to the body, while background
    false positives sit off on their own - so proximity, not size, is the right
    test, and it works even when a part is small because the person is far away.
    """
    m = (m > 0).astype(np.uint8)
    lbl, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lbl, range(1, n + 1))
    main = 1 + int(np.argmax(sizes))
    near = cv2.dilate((lbl == main).astype(np.uint8), _ellipse(dist)).astype(bool)
    keep = (lbl == main)
    for c in range(1, n + 1):
        if c == main:
            continue
        if sizes[c - 1] >= min_abs and (near & (lbl == c)).any():
            keep |= (lbl == c)
    return ndimage.binary_fill_holes(keep).astype(np.uint8)


def _clean(raw, close_ksize, keep_dist, open_ksize=3):
    """Noisy mask -> the person's foreground region (denoise, bridge, keep body parts)."""
    m = cv2.morphologyEx((raw > 0).astype(np.uint8), cv2.MORPH_OPEN, _ellipse(open_ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(close_ksize))
    return _keep_person(m, keep_dist)


def _colour_map_refine(frame, mask, erode_fg, dilate_bg, keep_dist, bins=32):
    """
    Section-8 colour-model + MAP refinement of the mask boundary.
    Confident foreground = interior of the mask (eroded); confident background =
    well outside it (dilated, inverted); the band between is uncertain. Build a
    foreground and a background colour histogram from the confident regions of this
    frame, then label each uncertain pixel by the MAP rule (higher-likelihood model
    wins).
    """
    sure_fg = cv2.erode(mask, _ellipse(erode_fg)).astype(bool)
    sure_bg = ~cv2.dilate(mask, _ellipse(dilate_bg)).astype(bool)
    unknown = (~sure_fg) & (~sure_bg)
    if sure_fg.sum() < 500 or sure_bg.sum() < 500:
        return mask

    shift = 256 // bins
    q = (frame // shift).astype(np.int32)
    idx = (q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]     # colour -> histogram bin
    h_fg = np.bincount(idx[sure_fg], minlength=bins ** 3).astype(np.float32)
    h_bg = np.bincount(idx[sure_bg], minlength=bins ** 3).astype(np.float32)
    h_fg /= h_fg.sum() + 1e-9
    h_bg /= h_bg.sum() + 1e-9

    fg = sure_fg | (unknown & (h_fg[idx] > h_bg[idx]))          # MAP decision
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, _ellipse(3))
    return _keep_person(fg, keep_dist)


def _temporal_fill(masks, keep_dist):
    """
    Recall-safe temporal smoothing: add any pixel that is foreground in BOTH the
    previous and next frame (recovers a one-frame drop-out) and never delete
    anything, so fast-moving parts survive.
    """
    out = []
    n = len(masks)
    for i in range(n):
        m = masks[i].copy()
        if 0 < i < n - 1:
            m = (m | (masks[i - 1] & masks[i + 1])).astype(np.uint8)
        out.append(_keep_person(m, keep_dist))
    return out


def run(input_path, transforms, meta, extracted_path, binary_path,
        dist2=300.0, fourcc='XVID'):
    """
    Build the KNN background model, classify + colour-refine every stabilized frame,
    steady the masks in time, and write the extracted (colour) and binary videos.
    Returns the list of foreground masks.
    """
    n, fps, w, h = meta
    # dist2Threshold is the colour-distance cutoff for "same as background". It only
    # sets the rough mask - the colour-model refinement re-decides the boundary from
    # colour each frame - so its exact value isn't critical. This is the one tuned knob.
    sub = cv2.createBackgroundSubtractorKNN(history=n, dist2Threshold=dist2,
                                            detectShadows=True)

    # pass 1: warm up the model, and (once warm, second half) measure the person's
    # HEIGHT so all the sizes below scale to the person rather than being fixed pixels.
    heights = []
    for i, (frame, valid) in enumerate(stab.warp_frames(input_path, transforms)):
        fg = sub.apply(frame)
        if i >= n // 2:
            m = cv2.morphologyEx(((fg == 255) & valid).astype(np.uint8),
                                 cv2.MORPH_OPEN, _ellipse(3))
            lbl, nc = ndimage.label(m)
            if nc > 0:
                big = 1 + int(np.argmax(ndimage.sum(m, lbl, range(1, nc + 1))))
                ys = np.where(lbl == big)[0]
                heights.append(ys.max() - ys.min())
    person_h = float(np.median(heights)) if heights else 671.0
    close_k = _odd(person_h * 0.016, 5)     # bridge small gaps
    erode_fg = _odd(person_h * 0.030, 7)    # confident-foreground shrink
    dilate_bg = _odd(person_h * 0.045, 9)   # confident-background band
    keep_dist = _odd(person_h * 0.080, 9)   # how close a detached part must be to count as body
    connect_k = _odd(person_h * 0.022, 9)   # close detection-break gaps (neck/knee) into one region

    # pass 2: classify, clean, and colour-refine each frame
    masks = []
    for frame, valid in stab.warp_frames(input_path, transforms):
        fg = sub.apply(frame)                  # 255 = foreground, 127 = shadow, 0 = background
        m = _clean(((fg == 255) & valid).astype(np.uint8), close_k, keep_dist)
        m = _colour_map_refine(frame, m, erode_fg, dilate_bg, keep_dist)
        # join the kept parts (head/leg detected separately) into one solid region so the
        # extracted person has no gaps; the kernel is small enough to leave the natural
        # gap between spread legs open.
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(connect_k))
        masks.append(ndimage.binary_fill_holes(m).astype(np.uint8))

    masks = _temporal_fill(masks, keep_dist)

    # pass 3: write, pairing each stabilized frame with its foreground mask
    ex_writer = cv2.VideoWriter(extracted_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    bin_writer = cv2.VideoWriter(binary_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    for i, (frame, _valid) in enumerate(stab.warp_frames(input_path, transforms)):
        m = masks[i]
        extracted = frame.copy()
        extracted[m == 0] = 0                  # foreground keeps its real colours, rest goes black
        ex_writer.write(extracted)
        # binary as 3-channel 0/255 so the colour codec keeps it; matting thresholds it back
        bin_writer.write(cv2.cvtColor(m * 255, cv2.COLOR_GRAY2BGR))
    ex_writer.release()
    bin_writer.release()
    return masks