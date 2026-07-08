"""
Background subtraction on the stabilized video.

This is the per-pixel background modelling from Section 8 of the notes: each pixel
is described by what it usually looks like over time, and a pixel is called
foreground when the current value disagrees with that background model. We use
OpenCV's KNN background subtractor (allowed for this project) as the model - a
non-parametric version of the same idea: rather than fitting Gaussians it keeps a
short history of recent samples per pixel and calls the pixel background when
enough of its nearest samples in that history are close in colour. Its shadow
detector flags pixels that are only a darker copy of the background (same colour,
lower brightness), which we drop so the floor shadow stays out of the foreground.

Because the clip is short and the camera is static after stabilization, we run two
passes: the first warms up the model over the whole clip so even early frames get
a good background, the second produces the foreground masks. Each mask is cleaned
into one connected foreground region (open -> close -> largest component -> fill
holes) and the stabilization border is forced to background. Finally a recall-safe
temporal step fills any pixel that is foreground in both neighbouring frames: this
recovers a limb that briefly dropped out without ever removing a real part, so
fast-moving arms and legs are kept.

Outputs:
  extracted_ID1_ID2.avi - foreground (person) in original colour, black elsewhere
  binary_ID1_ID2.avi    - foreground = 1 (stored as 255), background = 0
"""

import cv2
import numpy as np
from scipy import ndimage

import stabilization as stab


def _clean(raw, open_ksize=3, close_ksize=11):
    """Noisy mask -> one connected foreground region: open, close, largest blob, fill holes."""
    m = (raw > 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)))
    lbl, n = ndimage.label(m)
    if n > 0:
        sizes = ndimage.sum(m, lbl, range(1, n + 1))
        m = (lbl == (1 + int(np.argmax(sizes)))).astype(np.uint8)  # biggest blob = the person
    return ndimage.binary_fill_holes(m).astype(np.uint8)


def _temporal_fill(masks):
    """
    Recall-safe temporal smoothing: add any pixel that is foreground in BOTH the
    previous and next frame. This restores brief drop-outs (a limb blinking out for
    one frame) but never deletes anything, so fast-moving parts are preserved.
    """
    out = []
    n = len(masks)
    for i in range(n):
        m = masks[i].copy()
        if 0 < i < n - 1:
            m = (m | (masks[i - 1] & masks[i + 1])).astype(np.uint8)
        out.append(ndimage.binary_fill_holes(m).astype(np.uint8))
    return out


def run(input_path, transforms, meta, extracted_path, binary_path,
        dist2=300.0, fourcc='XVID'):
    """
    Build the KNN background model, classify every stabilized frame into
    foreground/background, and write the extracted (colour) and binary videos.
    Returns the list of foreground masks.
    """
    n, fps, w, h = meta
    # dist2Threshold is the colour-distance cutoff for "same as background" - the main
    # knob if a new video's person contrasts differently with its background.
    sub = cv2.createBackgroundSubtractorKNN(history=n, dist2Threshold=dist2,
                                            detectShadows=True)

    # pass 1: warm up the model on the whole clip (offline, static camera)
    for frame, _valid in stab.warp_frames(input_path, transforms):
        sub.apply(frame)

    # pass 2: classify + clean each frame
    masks = []
    for frame, valid in stab.warp_frames(input_path, transforms):
        fg = sub.apply(frame)                  # 255 = foreground, 127 = shadow, 0 = background
        masks.append(_clean(((fg == 255) & valid).astype(np.uint8)))   # drop shadow + border

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