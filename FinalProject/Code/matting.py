"""
Geodesic video matting.

Turns a stabilized color frame and a binary person mask into a soft opacity map
and a composite of the person over a new background.
"""

from __future__ import annotations

import heapq
import numpy as np
from cv2 import (
    MORPH_ELLIPSE,
    INTER_LINEAR,
    INTER_NEAREST,
    getStructuringElement,
    erode as cv_erode,
    dilate as cv_dilate,
    resize as cv_resize,
)
from dataclasses import dataclass
from scipy.ndimage import distance_transform_edt

BAND_RADIUS_PX = 9
SAMPLE_RING_PX = 4
BBOX_PAD_PX = 16

KDE_BANDWIDTH = 12.0
KDE_MAX_SAMPLES = 400
KDE_CHUNK_PX = 8192

GEODESIC_COLOR_WEIGHT = 1.0
GEODESIC_SPATIAL_WEIGHT = 0.02
ALPHA_DISTANCE_POWER = 1.0
ALPHA_SCALE = 2         # downscale factor for the geodesic/KDE computation

EPS = 1e-8

# (drow, dcol, step) for the 8-neighborhood
_NEIGHBORS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 2.0 ** 0.5), (-1, 1, 2.0 ** 0.5),
    (1, -1, 2.0 ** 0.5), (1, 1, 2.0 ** 0.5),
)


@dataclass(frozen=True)
class Trimap:
    """
    Trimap of a single frame.

    Attributes:
        fg: Confident-foreground mask (alpha == 1), full frame size.
        bg: Confident-background mask (alpha == 0), full frame size.
        band: Undecided band around the silhouette (alpha == phi), full frame size.
        bbox: (row0, row1, col0, col1) bounding box of the band, used to crop the
            frame for the per-band computation.
    """

    fg: np.ndarray
    bg: np.ndarray
    band: np.ndarray
    bbox: tuple[int, int, int, int]


def _disk(radius_px: int) -> np.ndarray:
    size = 2 * radius_px + 1
    return getStructuringElement(MORPH_ELLIPSE, (size, size))


def build_trimap(
    mask: np.ndarray,
    *,
    band_radius_px: int = BAND_RADIUS_PX,
    bbox_pad_px: int = BBOX_PAD_PX,
) -> Trimap:
    """
    Split a binary person mask into confident FG/BG and an undecided band.

    The band is the set of pixels within band_radius_px of the mask silhouette.

    Args:
        mask: 2-D boolean (or 0/1) array, True where the person is.
        band_radius_px: Half-width of the undecided band around the silhouette.
        bbox_pad_px: Extra padding added around the band bounding box.

    Returns:
        The Trimap for this frame.
    """
    mask_u8 = (mask > 0).astype(np.uint8)

    inner = cv_erode(mask_u8, _disk(band_radius_px)).astype(bool)
    outer = cv_dilate(mask_u8, _disk(band_radius_px)).astype(bool)
    band = outer & ~inner
    fg = inner          # alpha == 1, borders the band on the inside
    bg = ~outer         # alpha == 0, borders the band on the outside

    rows = np.flatnonzero(band.any(axis=1))
    cols = np.flatnonzero(band.any(axis=0))
    if rows.size == 0:
        return Trimap(fg, bg, band, (0, 0, 0, 0))

    h, w = mask.shape
    row0 = max(int(rows[0]) - bbox_pad_px, 0)
    row1 = min(int(rows[-1]) + bbox_pad_px + 1, h)
    col0 = max(int(cols[0]) - bbox_pad_px, 0)
    col1 = min(int(cols[-1]) + bbox_pad_px + 1, w)
    return Trimap(fg, bg, band, (row0, row1, col0, col1))


def _subsample(colors: np.ndarray, max_samples: int) -> np.ndarray:
    if colors.shape[0] <= max_samples:
        return colors
    idx = np.random.default_rng(0).choice(colors.shape[0], max_samples, replace=False)
    return colors[idx]


def _parzen_density(query: np.ndarray, samples: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Gaussian Parzen density f(c) = mean_j exp(-||c - s_j||^2 / 2 sigma^2).
    """
    if samples.shape[0] == 0:
        return np.zeros(query.shape[0])

    inv_two_sigma_sq = 1.0 / (2.0 * bandwidth * bandwidth)
    density = np.empty(query.shape[0])
    for start in range(0, query.shape[0], KDE_CHUNK_PX):
        chunk = query[start : start + KDE_CHUNK_PX]
        sq_dist = np.sum((chunk[:, None, :] - samples[None, :, :]) ** 2, axis=2)
        density[start : start + chunk.shape[0]] = np.mean(np.exp(-sq_dist * inv_two_sigma_sq), axis=1)
    return density


def _foreground_posterior(
    colors: np.ndarray,
    region: np.ndarray,
    fg_samples: np.ndarray,
    bg_samples: np.ndarray,
) -> np.ndarray:
    """
    Posterior P(F|c) = f(c|F) / (f(c|F) + f(c|B)) with equal priors, over `region`.
    """
    prob = np.zeros(colors.shape[:2])
    query = colors[region].astype(np.float64)
    f_fg = _parzen_density(query, fg_samples, KDE_BANDWIDTH)
    f_bg = _parzen_density(query, bg_samples, KDE_BANDWIDTH)
    prob[region] = f_fg / (f_fg + f_bg + EPS)
    return prob


def _geodesic_distance(prob_fg: np.ndarray, seeds: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """
    Geodesic distance from `seeds` to every allowed pixel via Dijkstra.

    The edge weight between neighbors a, b penalizes crossing a color edge:

        w(a, b) = color_weight * |P_F(a) - P_F(b)| + spatial_weight * step

    Args:
        prob_fg: Foreground posterior P(F|c) over the crop.
        seeds: Pixels with distance 0 (sources).
        allowed: Pixels the paths may traverse.

    Returns:
        Distance array; np.inf where unreachable / outside `allowed`.
    """
    h, w = prob_fg.shape
    dist = np.full((h, w), np.inf)
    visited = np.zeros((h, w), dtype=bool)

    heap = []
    seed_rows, seed_cols = np.nonzero(seeds & allowed)
    for r, c in zip(seed_rows.tolist(), seed_cols.tolist()):
        dist[r, c] = 0.0
        heap.append((0.0, r, c))
    heapq.heapify(heap)

    while heap:
        d, r, c = heapq.heappop(heap)
        if visited[r, c]:
            continue
        visited[r, c] = True
        p_here = prob_fg[r, c]
        for dr, dc, step in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if not allowed[nr, nc] or visited[nr, nc]:
                continue
            nd = d + GEODESIC_COLOR_WEIGHT * abs(p_here - prob_fg[nr, nc]) + GEODESIC_SPATIAL_WEIGHT * step
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                heapq.heappush(heap, (nd, nr, nc))
    return dist


def estimate_alpha(
    frame_bgr: np.ndarray,
    trimap: Trimap,
    *,
    distance_power: float = ALPHA_DISTANCE_POWER,
) -> np.ndarray:
    """
    Estimate the opacity map alpha in [0, 1] for a single frame.

    Confident FG gets 1, confident BG gets 0, and band pixels get
    alpha = w_F / (w_F + w_B) with w_F = D_F^{-r} * P_F. The geodesic/KDE work runs
    on a downscaled crop for speed and the band alpha is upsampled back.

    Args:
        frame_bgr: H x W x 3 BGR frame.
        trimap: Trimap for this frame.
        distance_power: Exponent r applied to the geodesic distances, r in (0, 2].

    Returns:
        H x W float32 opacity map in [0, 1].
    """
    alpha = trimap.fg.astype(np.float32)
    row0, row1, col0, col1 = trimap.bbox
    if row1 <= row0 or col1 <= col0:
        return alpha

    band_c = trimap.band[row0:row1, col0:col1]
    h_c, w_c = band_c.shape

    # The geodesic/KDE part is the bottleneck, so run it on a downscaled crop and
    # upsample the band alpha back; confident FG/BG stay crisp from the full-res mask.
    hs, ws = max(h_c // ALPHA_SCALE, 1), max(w_c // ALPHA_SCALE, 1)
    color_s = cv_resize(frame_bgr[row0:row1, col0:col1], (ws, hs), interpolation=INTER_LINEAR).astype(np.float32)
    band_s = cv_resize(band_c.astype(np.uint8), (ws, hs), interpolation=INTER_NEAREST).astype(bool)
    fg_s = cv_resize(trimap.fg[row0:row1, col0:col1].astype(np.uint8), (ws, hs), interpolation=INTER_NEAREST).astype(bool)
    bg_s = cv_resize(trimap.bg[row0:row1, col0:col1].astype(np.uint8), (ws, hs), interpolation=INTER_NEAREST).astype(bool)

    # ring: band + 1px so the seeds just inside FG / outside BG carry a P_F value;
    # sample_region: a few px more, for richer boundary color samples.
    ring = cv_dilate(band_s.astype(np.uint8), _disk(1)).astype(bool)
    sample_region = cv_dilate(band_s.astype(np.uint8), _disk(SAMPLE_RING_PX)).astype(bool)
    fg_samples = _subsample(color_s[fg_s & sample_region].astype(np.float64), KDE_MAX_SAMPLES)
    bg_samples = _subsample(color_s[bg_s & sample_region].astype(np.float64), KDE_MAX_SAMPLES)
    if fg_samples.shape[0] == 0 or bg_samples.shape[0] == 0:
        return alpha

    prob_fg = _foreground_posterior(color_s, ring, fg_samples, bg_samples)
    dist_fg = _geodesic_distance(prob_fg, fg_s & ring, ring)
    dist_bg = _geodesic_distance(prob_fg, bg_s & ring, ring)

    p_fg = np.clip(prob_fg[band_s], EPS, 1.0)
    p_bg = np.clip(1.0 - prob_fg[band_s], EPS, 1.0)
    d_fg = np.maximum(dist_fg[band_s], EPS)
    d_bg = np.maximum(dist_bg[band_s], EPS)

    w_fg = d_fg ** (-distance_power) * p_fg
    w_bg = d_bg ** (-distance_power) * p_bg
    band_alpha = np.where(np.isfinite(w_fg) & np.isfinite(w_bg), w_fg / (w_fg + w_bg + EPS), 1.0)

    alpha_s = fg_s.astype(np.float32)
    alpha_s[band_s] = band_alpha.astype(np.float32)
    alpha_up = cv_resize(alpha_s, (w_c, h_c), interpolation=INTER_LINEAR)

    crop = alpha[row0:row1, col0:col1]
    crop[band_c] = alpha_up[band_c]
    alpha[row0:row1, col0:col1] = crop
    return np.clip(alpha, 0.0, 1.0)


def estimate_foreground_color(frame_bgr: np.ndarray, trimap: Trimap) -> np.ndarray:
    """
    Estimate the pure foreground color F(x) for every pixel.

    Instead of the window search that solves c = alpha*F + (1-alpha)*B for F, we use
    the equivalent-in-spirit approximation: propagate the color of the nearest
    confident-foreground pixel into the band, which removes the background color halo.

    Args:
        frame_bgr: H x W x 3 BGR frame.
        trimap: Trimap for this frame.

    Returns:
        H x W x 3 foreground-color image, same dtype as frame_bgr.
    """
    fg_color = frame_bgr.copy()
    row0, row1, col0, col1 = trimap.bbox
    if row1 <= row0 or col1 <= col0:
        return fg_color

    fg_c = trimap.fg[row0:row1, col0:col1]
    _, (idx_r, idx_c) = distance_transform_edt(~fg_c, return_indices=True)
    fg_color[row0:row1, col0:col1] = frame_bgr[row0:row1, col0:col1][idx_r, idx_c]
    return fg_color


def composite(
    alpha: np.ndarray,
    foreground_bgr: np.ndarray,
    background_bgr: np.ndarray,
) -> np.ndarray:
    """
    Blend the estimated foreground onto the new background: J = a*F + (1-a)*B.

    Args:
        alpha: H x W opacity map in [0, 1].
        foreground_bgr: H x W x 3 estimated foreground colors.
        background_bgr: H x W x 3 new background, already resized to the frame size.

    Returns:
        H x W x 3 matted frame, uint8.
    """
    a = alpha[:, :, None].astype(np.float32)
    matted = a * foreground_bgr.astype(np.float32) + (1.0 - a) * background_bgr.astype(np.float32)
    return np.clip(matted, 0, 255).astype(np.uint8)


def matte_frame(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    background_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the full matting pipeline on a single frame.

    Args:
        frame_bgr: H x W x 3 BGR stabilized frame.
        mask: H x W binary person mask (True/1 = person).
        background_bgr: New background, already resized to the frame size.

    Returns:
        (matted_bgr uint8, alpha float32 in [0, 1]).
    """
    trimap = build_trimap(mask)
    alpha = estimate_alpha(frame_bgr, trimap)
    foreground = estimate_foreground_color(frame_bgr, trimap)
    return composite(alpha, foreground, background_bgr), alpha


def matte_video(
    frames_bgr: list[np.ndarray],
    masks: list[np.ndarray],
    background_bgr: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Matte a whole clip.

    Args:
        frames_bgr: Stabilized BGR frames.
        masks: Binary person masks, one per frame.
        background_bgr: New background image (any size; resized to the frame size).

    Returns:
        (matted_frames, alpha_frames) as parallel lists.

    Raises:
        ValueError: If frames and masks have different lengths.
    """
    if len(frames_bgr) != len(masks):
        raise ValueError(f"frames/masks length mismatch: {len(frames_bgr)} vs {len(masks)}")

    h, w = frames_bgr[0].shape[:2]
    background_resized = cv_resize(background_bgr, (w, h), interpolation=INTER_LINEAR)

    matted_frames, alpha_frames = [], []
    for frame, mask in zip(frames_bgr, masks):
        matted, alpha = matte_frame(frame, mask, background_resized)
        matted_frames.append(matted)
        alpha_frames.append(alpha)
    return matted_frames, alpha_frames


MASK_THRESHOLD = 127     # binary.avi is expected as 0/255 (person = 255)
VIDEO_FOURCC = "XVID"


def _binary_frame_to_mask(frame: np.ndarray) -> np.ndarray:
    """
    Turn one binary-video frame into a boolean person mask.

    Handles a single- or 3-channel frame; person pixels are above MASK_THRESHOLD.
    """
    if frame.ndim == 3:
        frame = frame[:, :, 0]
    return frame > MASK_THRESHOLD


def run_matting(
    stabilize_path: str,
    binary_path: str,
    background_path: str,
    matted_path: str,
    alpha_path: str,
) -> int:
    """
    Read stabilize.avi + binary.avi, matte over the background, write the outputs.

    Frames are streamed one at a time, so memory stays flat regardless of clip length.
    alpha is written as a 0..255 video (round(alpha * 255), replicated to 3 channels).

    Args:
        stabilize_path: Path to the stabilized color video.
        binary_path: Path to the binary mask video (0/255, person = 255).
        background_path: Path to the new background image.
        matted_path: Where to write the matted video.
        alpha_path: Where to write the alpha video.

    Returns:
        Number of frames written.

    Raises:
        FileNotFoundError: If an input video or the background cannot be opened.
    """
    import cv2  # imported here so the library has no import-time cv2-I/O dependency

    stab = cv2.VideoCapture(stabilize_path)
    binv = cv2.VideoCapture(binary_path)
    background = cv2.imread(background_path)
    if not stab.isOpened():
        raise FileNotFoundError(f"cannot open stabilized video: {stabilize_path}")
    if not binv.isOpened():
        raise FileNotFoundError(f"cannot open binary video: {binary_path}")
    if background is None:
        raise FileNotFoundError(f"cannot read background image: {background_path}")

    width = int(stab.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stab.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = stab.get(cv2.CAP_PROP_FPS) or 30.0
    background = cv2.resize(background, (width, height))

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
    matted_writer = cv2.VideoWriter(matted_path, fourcc, fps, (width, height))
    alpha_writer = cv2.VideoWriter(alpha_path, fourcc, fps, (width, height))

    count = 0
    while True:
        ok_stab, frame = stab.read()
        ok_bin, mask_frame = binv.read()
        if not ok_stab or not ok_bin:
            break
        mask = _binary_frame_to_mask(mask_frame)
        matted, alpha = matte_frame(frame, mask, background)
        matted_writer.write(matted)
        alpha_writer.write(cv2.cvtColor((alpha * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        count += 1

    stab.release()
    binv.release()
    matted_writer.release()
    alpha_writer.release()
    return count
