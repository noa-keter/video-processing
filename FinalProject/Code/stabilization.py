"""
Feature-based video stabilization with a 2D similarity motion model.

Track corners between frames to estimate how the camera moved, build the camera
path over the clip, flatten it (the camera is meant to be still), and warp each
frame back so the background stops shaking. OpenCV is only used for the Harris
detector and the LK tracker; the fit, RANSAC, trajectory and smoothing are ours.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def fit_similarity_lsq(pts0, pts1):
    """
    Least-squares 2D similarity mapping pts0 -> pts1. The model
        x' = a*x - b*y + tx,  y' = b*x + a*y + ty   (a = s*cos, b = s*sin)
    is linear in (a, b, tx, ty), so we stack all points and solve. None if n < 2.
    """
    n = len(pts0)
    if n < 2:
        return None
    x, y = pts0[:, 0], pts0[:, 1]
    xp, yp = pts1[:, 0], pts1[:, 1]

    # each point -> two rows: even row is its x' eq, odd row its y' eq
    A = np.zeros((2 * n, 4), dtype=np.float64)
    A[0::2, 0] = x;   A[0::2, 1] = -y;  A[0::2, 2] = 1.0
    A[1::2, 0] = y;   A[1::2, 1] = x;   A[1::2, 3] = 1.0
    c = np.empty(2 * n, dtype=np.float64)
    c[0::2] = xp
    c[1::2] = yp

    p, *_ = np.linalg.lstsq(A, c, rcond=None)
    return p  # (a, b, tx, ty)


def _apply_similarity(p, pts):
    a, b, tx, ty = p
    xp = a * pts[:, 0] - b * pts[:, 1] + tx
    yp = b * pts[:, 0] + a * pts[:, 1] + ty
    return np.stack([xp, yp], axis=1)


def ransac_similarity(pts0, pts1, thresh=3.0, iters=200, rng=None):
    """
    RANSAC around the similarity fit so bad matches (mostly the moving person)
    don't skew it. Two pairs pin down a similarity; keep the largest inlier set
    and refit on it. Returns (params, inlier_mask).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(pts0)
    if n < 2:
        return None, None

    best_inliers, best_count = None, -1
    idx_all = np.arange(n)
    for _ in range(iters):
        sample = rng.choice(idx_all, size=2, replace=False)
        p = fit_similarity_lsq(pts0[sample], pts1[sample])
        if p is None:
            continue
        err = np.linalg.norm(_apply_similarity(p, pts0) - pts1, axis=1)
        inliers = err < thresh
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count, best_inliers = cnt, inliers

    if best_inliers is None or best_count < 2:
        return fit_similarity_lsq(pts0, pts1), np.ones(n, bool)  # nothing agreed, fit all
    return fit_similarity_lsq(pts0[best_inliers], pts1[best_inliers]), best_inliers


# Harris response (useHarrisDetector=True, k = alpha). minDistance spreads corners
# so motion is sampled across the whole frame, not just on the posters.
_FEATURE_PARAMS = dict(maxCorners=800, qualityLevel=0.01, minDistance=8, blockSize=7,
                       useHarrisDetector=True, k=0.04)
# maxLevel=3 pyramid so LK still catches the bigger shakes (~9 px measured).
_LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def estimate_pair_motion(prev_gray, cur_gray, rng):
    """
    Camera motion between two frames as (dx, dy, rotation), or None on failure.
    Scale is fitted but dropped (< 0.1% here) so the person can't get stretched.
    """
    p0 = cv2.goodFeaturesToTrack(prev_gray, **_FEATURE_PARAMS)
    if p0 is None or len(p0) < 8:
        return None

    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None, **_LK_PARAMS)
    # forward-backward check: track back and keep only points that return home
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, p1, None, **_LK_PARAMS)
    fb_err = np.linalg.norm(p0 - p0r, axis=2).reshape(-1)
    good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb_err < 1.0)
    if good.sum() < 8:
        good = (st.reshape(-1) == 1)  # too strict, fall back to forward tracks

    a0, a1 = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
    if len(a0) < 8:
        return None

    p, _ = ransac_similarity(a0, a1, thresh=3.0, iters=200, rng=rng)
    if p is None:
        return None
    a, b, tx, ty = p
    return float(tx), float(ty), float(np.arctan2(b, a))


def smooth_trajectory(trajectory, sigma=25.0, static_camera=True):
    """
    Target camera path. Default: camera is static, so pin every frame to the mean
    position - removes jitter and drift together (what background subtraction needs)
    and keeps borders smallest. static_camera=False Gaussian-smooths instead.
    """
    smoothed = np.empty_like(trajectory)
    if static_camera:
        smoothed[:] = trajectory.mean(axis=0, keepdims=True)
    else:
        for k in range(trajectory.shape[1]):
            smoothed[:, k] = gaussian_filter1d(trajectory[:, k], sigma=sigma, mode='nearest')
    return smoothed


def _build_affine(dx, dy, da):
    ca, sa = np.cos(da), np.sin(da)
    return np.array([[ca, -sa, dx], [sa, ca, dy]], dtype=np.float64)


def compute_stabilizing_transforms(input_path, smoothing_sigma=25.0, static_camera=True):
    """
    Pass 1: estimate per-frame motion, accumulate + smooth the path, and return one
    correction matrix per frame plus (n_frames, fps, w, h).
    """
    cap = cv2.VideoCapture(input_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rng = np.random.default_rng(0)  # fixed seed -> reproducible RANSAC
    ok, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    motions, last = [], (0.0, 0.0, 0.0)
    while True:
        ok, cur = cap.read()
        if not ok:
            break
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        m = estimate_pair_motion(prev_gray, cur_gray, rng)
        if m is None:
            m = last  # reuse last motion so the path stays continuous
        motions.append(m)
        last = m
        prev_gray = cur_gray
    cap.release()

    motions = np.array(motions, dtype=np.float64)
    trajectory = np.zeros((n, 3), dtype=np.float64)   # absolute path, frame 0 at origin
    trajectory[1:] = np.cumsum(motions, axis=0)
    smoothed = smooth_trajectory(trajectory, sigma=smoothing_sigma, static_camera=static_camera)
    diff = smoothed - trajectory                       # per-frame shift onto the smooth path

    transforms = [_build_affine(diff[i, 0], diff[i, 1], diff[i, 2]) for i in range(n)]
    return transforms, (n, fps, w, h)


def warp_frames(input_path, transforms):
    """
    Pass 2 (generator): apply each correction, yielding (stabilized_bgr, valid_mask).
    valid_mask marks real pixels vs. the black border, for background subtraction.
    """
    cap = cv2.VideoCapture(input_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ones = np.full((h, w), 255, np.uint8)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        M = transforms[i]
        warped = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # nearest-neighbour keeps the mask a clean yes/no (no grey border bleed)
        valid = cv2.warpAffine(ones, M, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        yield warped, valid
        i += 1
    cap.release()


def stabilize(input_path, output_path, smoothing_sigma=25.0, static_camera=True, fourcc='XVID'):
    """
    End to end: run both passes, write the stabilized video, return (transforms, meta).
    XVID so the .avi opens in Windows; OpenCV reads it back either way.
    """
    transforms, (n, fps, w, h) = compute_stabilizing_transforms(
        input_path, smoothing_sigma, static_camera)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    for warped, _valid in warp_frames(input_path, transforms):
        writer.write(warped)
    writer.release()
    return transforms, (n, fps, w, h)