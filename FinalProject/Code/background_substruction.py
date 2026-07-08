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

2. Colour-model refinement (also from Section 8: a mixture/likelihood model with a
   MAP decision). The KNN mask is confident deep inside the person and far outside,
   but rough at the boundary. So we build a foreground colour model and a
   background colour model from those confident regions of the CURRENT frame, and
   for the uncertain band in between we decide each pixel by the MAP rule from the
   notes - assign it to whichever model gives the higher likelihood. The colour
   models are simple histograms (a non-parametric density estimate) rebuilt every
   frame, so they adapt to the person's clothes and the scene instead of relying
   on any fixed colours (this keeps it from overfitting one video).

The camera is static after stabilization, so we run two passes: the first warms up
the KNN model over the whole clip, the second produces the masks. A final
recall-safe temporal step fills any pixel that is foreground in both neighbouring
frames, recovering a limb that briefly dropped out without ever deleting a real
part.

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


def _largest_filled(m):
    """Keep the biggest connected blob (the person) and fill its holes."""
    m = (m > 0).astype(np.uint8)
    lbl, n = ndimage.label(m)
    if n > 0:
        sizes = ndimage.sum(m, lbl, range(1, n + 1))
        m = (lbl == (1 + int(np.argmax(sizes)))).astype(np.uint8)
    return ndimage.binary_fill_holes(m).astype(np.uint8)


def _clean(raw, close_ksize, open_ksize=3):
    """Noisy mask -> one connected foreground region."""
    m = cv2.morphologyEx((raw > 0).astype(np.uint8), cv2.MORPH_OPEN, _ellipse(open_ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(close_ksize))
    return _largest_filled(m)


def _colour_map_refine(frame, mask, erode_fg, dilate_bg, bins=32):
    """
    Section-8 colour-model + MAP refinement of the mask boundary.
    Confident foreground = interior of the mask (eroded); confident background =
    well outside it (dilated, inverted); the band between is uncertain. Build a
    foreground and a background colour histogram from the confident regions of this
    frame, then label each uncertain pixel by the MAP rule: whichever model gives
    the higher likelihood wins.
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
    return _largest_filled(fg)


def _temporal_fill(masks):
    """
    Recall-safe temporal smoothing: add any pixel that is foreground in BOTH the
    previous and next frame (recovers a one-frame drop-out) and never delete
    anything, so fast-moving parts survive. A final largest-component keeps the
    result a single clean region.
    """
    out = []
    n = len(masks)
    for i in range(n):
        m = masks[i].copy()
        if 0 < i < n - 1:
            m = (m | (masks[i - 1] & masks[i + 1])).astype(np.uint8)
        out.append(_largest_filled(m))
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
    # colour each frame, so the exact value isn't critical. This is the one tuned knob.
    sub = cv2.createBackgroundSubtractorKNN(history=n, dist2Threshold=dist2,
                                            detectShadows=True)

    # pass 1: warm up the model on the whole clip, and (once it's warm, second half)
    # measure the person's HEIGHT so the morphology scales to the person's size rather
    # than using pixel counts tuned to one video. Height is used because it's stable
    # and not inflated by horizontal noise the way the width is.
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
    # fractions of the person's height (they reproduce 11 / 21 / 31 at height ~671)
    close_k = _odd(person_h * 0.016, 5)
    erode_fg = _odd(person_h * 0.030, 7)
    dilate_bg = _odd(person_h * 0.045, 9)

    # pass 2: classify, clean, and colour-refine each frame
    masks = []
    for frame, valid in stab.warp_frames(input_path, transforms):
        fg = sub.apply(frame)                  # 255 = foreground, 127 = shadow, 0 = background
        m = _clean(((fg == 255) & valid).astype(np.uint8), close_k)   # drop shadow + border
        masks.append(_colour_map_refine(frame, m, erode_fg, dilate_bg))

    masks = _temporal_fill(masks)

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