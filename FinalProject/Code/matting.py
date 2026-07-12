"""
Geodesic video matting.

Turns a stabilized color frame and a binary person mask into a soft opacity map
and a composite of the person over a new background. The mask is trusted away
from the silhouette; only a thin band around the edge is re-decided, with a
color model (Parzen KDE) plus geodesic distances from each side of the band.
"""

import heapq
from collections import namedtuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

BAND_RADIUS_PX = 9      # half-width of the undecided band around the silhouette
SAMPLE_RING_PX = 4      # how far past the band boundary color samples are taken
BBOX_PAD_PX = 16

KDE_BANDWIDTH = 12.0
KDE_MAX_SAMPLES = 400
KDE_CHUNK_PX = 8192

GEODESIC_COLOR_WEIGHT = 1.0
GEODESIC_SPATIAL_WEIGHT = 0.02
ALPHA_DISTANCE_POWER = 1.0
ALPHA_SCALE = 2         # downscale factor for the geodesic/KDE computation

MASK_THRESHOLD = 127    # binary.avi is 0/255 (person = 255)
VIDEO_FOURCC = "XVID"
EPS = 1e-8

# (drow, dcol, step) for the 8-neighborhood
_NEIGHBORS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 2.0 ** 0.5), (-1, 1, 2.0 ** 0.5),
    (1, -1, 2.0 ** 0.5), (1, 1, 2.0 ** 0.5),
)

# fg = confident foreground (alpha 1), bg = confident background (alpha 0),
# band = undecided ring between them, bbox = (row0, row1, col0, col1) crop
# around the band - all the per-frame work happens inside it.
Trimap = namedtuple("Trimap", ["fg", "bg", "band", "bbox"])


def _disk(radius_px):
    size = 2 * radius_px + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def build_trimap(mask, band_radius_px=BAND_RADIUS_PX, bbox_pad_px=BBOX_PAD_PX):
    """
    Split a binary person mask into confident fg/bg and the undecided band:
    everything within band_radius_px of the mask silhouette.
    """
    mask_u8 = (mask > 0).astype(np.uint8)

    inner = cv2.erode(mask_u8, _disk(band_radius_px)).astype(bool)
    outer = cv2.dilate(mask_u8, _disk(band_radius_px)).astype(bool)
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


def _subsample(colors, max_samples):
    if colors.shape[0] <= max_samples:
        return colors
    idx = np.random.default_rng(0).choice(colors.shape[0], max_samples, replace=False)
    return colors[idx]


def _parzen_density(query, samples, bandwidth):
    """Gaussian Parzen density f(c) = mean_j exp(-||c - s_j||^2 / 2 sigma^2)."""
    if samples.shape[0] == 0:
        return np.zeros(query.shape[0])

    inv_two_sigma_sq = 1.0 / (2.0 * bandwidth * bandwidth)
    density = np.empty(query.shape[0])
    for start in range(0, query.shape[0], KDE_CHUNK_PX):
        chunk = query[start : start + KDE_CHUNK_PX]
        sq_dist = np.sum((chunk[:, None, :] - samples[None, :, :]) ** 2, axis=2)
        density[start : start + chunk.shape[0]] = np.mean(np.exp(-sq_dist * inv_two_sigma_sq), axis=1)
    return density


def _foreground_posterior(colors, region, fg_samples, bg_samples):
    """Posterior P(F|c) = f(c|F) / (f(c|F) + f(c|B)) with equal priors, over `region`."""
    prob = np.zeros(colors.shape[:2])
    query = colors[region].astype(np.float64)
    f_fg = _parzen_density(query, fg_samples, KDE_BANDWIDTH)
    f_bg = _parzen_density(query, bg_samples, KDE_BANDWIDTH)
    prob[region] = f_fg / (f_fg + f_bg + EPS)
    return prob


def _geodesic_distance(prob_fg, seeds, allowed):
    """
    Geodesic distance from `seeds` to every `allowed` pixel via Dijkstra. The edge
    weight between neighbors penalizes crossing a color edge:
        w(a, b) = color_weight * |P_F(a) - P_F(b)| + spatial_weight * step
    so the distance follows object outlines instead of cutting through them.
    Unreachable pixels stay at np.inf.
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


def estimate_alpha(frame_bgr, trimap, distance_power=ALPHA_DISTANCE_POWER):
    """
    Opacity map alpha in [0, 1] for one frame. Confident fg gets 1, confident bg
    gets 0, and band pixels get alpha = w_F / (w_F + w_B) with w_F = D_F^{-r} * P_F.
    """
    alpha = trimap.fg.astype(np.float32)
    row0, row1, col0, col1 = trimap.bbox
    if row1 <= row0 or col1 <= col0:
        return alpha

    band_c = trimap.band[row0:row1, col0:col1]
    h_c, w_c = band_c.shape

    # The geodesic/KDE part is the bottleneck, so run it on a downscaled crop and
    # upsample the band alpha back; confident fg/bg stay crisp from the full-res mask.
    hs, ws = max(h_c // ALPHA_SCALE, 1), max(w_c // ALPHA_SCALE, 1)
    color_s = cv2.resize(frame_bgr[row0:row1, col0:col1], (ws, hs), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    band_s = cv2.resize(band_c.astype(np.uint8), (ws, hs), interpolation=cv2.INTER_NEAREST).astype(bool)
    fg_s = cv2.resize(trimap.fg[row0:row1, col0:col1].astype(np.uint8), (ws, hs), interpolation=cv2.INTER_NEAREST).astype(bool)
    bg_s = cv2.resize(trimap.bg[row0:row1, col0:col1].astype(np.uint8), (ws, hs), interpolation=cv2.INTER_NEAREST).astype(bool)

    # ring: band + 1px so the seeds just inside fg / outside bg carry a P_F value;
    # sample_region: a few px more, for richer boundary color samples.
    ring = cv2.dilate(band_s.astype(np.uint8), _disk(1)).astype(bool)
    sample_region = cv2.dilate(band_s.astype(np.uint8), _disk(SAMPLE_RING_PX)).astype(bool)
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
    alpha_up = cv2.resize(alpha_s, (w_c, h_c), interpolation=cv2.INTER_LINEAR)

    crop = alpha[row0:row1, col0:col1]
    crop[band_c] = alpha_up[band_c]
    alpha[row0:row1, col0:col1] = crop
    return np.clip(alpha, 0.0, 1.0)


def estimate_foreground_color(frame_bgr, trimap):
    """
    Pure foreground color F(x) for every pixel. Instead of the window search that
    solves c = alpha*F + (1-alpha)*B for F, propagate the color of the nearest
    confident-foreground pixel into the band - same goal (kills the background
    color halo), much cheaper.
    """
    fg_color = frame_bgr.copy()
    row0, row1, col0, col1 = trimap.bbox
    if row1 <= row0 or col1 <= col0:
        return fg_color

    fg_c = trimap.fg[row0:row1, col0:col1]
    _, (idx_r, idx_c) = distance_transform_edt(~fg_c, return_indices=True)
    fg_color[row0:row1, col0:col1] = frame_bgr[row0:row1, col0:col1][idx_r, idx_c]
    return fg_color


def composite(alpha, foreground_bgr, background_bgr):
    """Blend the estimated foreground onto the new background: J = a*F + (1-a)*B."""
    a = alpha[:, :, None].astype(np.float32)
    matted = a * foreground_bgr.astype(np.float32) + (1.0 - a) * background_bgr.astype(np.float32)
    return np.clip(matted, 0, 255).astype(np.uint8)


def matte_frame(frame_bgr, mask, background_bgr):
    """Full matting of one frame; returns (matted uint8, alpha float32 in [0, 1])."""
    trimap = build_trimap(mask)
    alpha = estimate_alpha(frame_bgr, trimap)
    foreground = estimate_foreground_color(frame_bgr, trimap)
    return composite(alpha, foreground, background_bgr), alpha


def _binary_frame_to_mask(frame):
    """One binary-video frame -> boolean person mask (single- or 3-channel input)."""
    if frame.ndim == 3:
        frame = frame[:, :, 0]
    return frame > MASK_THRESHOLD


def run_matting(stabilize_path, binary_path, background_path, matted_path, alpha_path):
    """
    Read stabilize.avi + binary.avi, matte every frame over the background image,
    write matted.avi and alpha.avi (alpha scaled to 0..255, 3-channel). Frames are
    streamed one at a time so memory stays flat. Returns the frame count.
    """
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
