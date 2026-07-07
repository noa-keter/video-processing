"""
Background subtraction on the stabilized video.

We model the background with OpenCV's KNN subtractor (allowed for this project).
It's a non-parametric version of the per-pixel background idea from the notes:
instead of fitting Gaussians, it keeps a short history of recent samples for each
pixel and calls the pixel background when enough of its nearest samples in that
history are close in colour - a kernel-density / nearest-neighbour take on the
same "what does this pixel usually look like" question. Its shadow detector flags
pixels that are just a darker version of the background (same colour, lower
brightness) and we drop those, which keeps the floor shadow out of the mask.

Because the video is short and the camera is static after stabilization, we make
two passes: the first warms up the model over the whole clip so even early frames
get a well-formed background, the second produces the masks. Each raw mask is then
cleaned into one solid person (open -> close -> largest component -> fill holes),
and the stabilization border is forced to background.

Outputs:
  extracted_ID1_ID2.avi - person in original colour, black everywhere else
  binary_ID1_ID2.avi    - person = 1 (stored as 255), background = 0
"""

import cv2
import numpy as np
from scipy import ndimage

import stabilization as stab


def _refine(raw, open_ksize=3, close_ksize=9):
    """Noisy mask -> one solid silhouette: open, close, keep largest blob, fill holes."""
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


def run(input_path, transforms, meta, extracted_path, binary_path,
        dist2=300.0, fourcc='XVID'):
    """
    Build the KNN background model, then classify every stabilized frame and write
    the extracted (colour) and binary videos. Returns the list of binary masks.
    """
    n, fps, w, h = meta
    # dist2Threshold is the colour-distance cutoff for "same as background" - the main
    # knob if a new video's person contrasts differently with its background.
    sub = cv2.createBackgroundSubtractorKNN(history=n, dist2Threshold=dist2,
                                            detectShadows=True)

    # pass 1: warm up the model on the whole clip (offline, static camera)
    for frame, _valid in stab.warp_frames(input_path, transforms):
        sub.apply(frame)

    ex_writer = cv2.VideoWriter(extracted_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    bin_writer = cv2.VideoWriter(binary_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

    # pass 2: classify and write
    masks = []
    for frame, valid in stab.warp_frames(input_path, transforms):
        fg = sub.apply(frame)                  # 255 = foreground, 127 = shadow, 0 = background
        m = _refine(((fg == 255) & valid).astype(np.uint8))   # drop shadow + stabilization border
        masks.append(m)

        extracted = frame.copy()
        extracted[m == 0] = 0                  # person keeps its real colours, rest goes black
        ex_writer.write(extracted)
        # binary as 3-channel 0/255 so the colour codec keeps it; matting thresholds it back
        bin_writer.write(cv2.cvtColor(m * 255, cv2.COLOR_GRAY2BGR))

    ex_writer.release()
    bin_writer.release()
    return masks