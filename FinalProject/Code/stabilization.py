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


def fit_similarity_lsq(src_points, dst_points):
    """
    Least-squares 2D similarity mapping src_points -> dst_points. The model
        x' = a*x - b*y + tx,  y' = b*x + a*y + ty   (a = s*cos, b = s*sin)
    is linear in (a, b, tx, ty), so we stack all points and solve. None if n < 2.
    """
    num_points = len(src_points)
    if num_points < 2:
        return None
    x, y = src_points[:, 0], src_points[:, 1]
    x_dst, y_dst = dst_points[:, 0], dst_points[:, 1]

    # each point -> two rows: even row is its x' eq, odd row its y' eq
    design_matrix = np.zeros((2 * num_points, 4), dtype=np.float64)
    design_matrix[0::2, 0] = x;   design_matrix[0::2, 1] = -y;  design_matrix[0::2, 2] = 1.0
    design_matrix[1::2, 0] = y;   design_matrix[1::2, 1] = x;   design_matrix[1::2, 3] = 1.0
    target = np.empty(2 * num_points, dtype=np.float64)
    target[0::2] = x_dst
    target[1::2] = y_dst

    params, *_ = np.linalg.lstsq(design_matrix, target, rcond=None)
    return params  # (a, b, tx, ty)


def _apply_similarity(params, points):
    a, b, tx, ty = params
    x_mapped = a * points[:, 0] - b * points[:, 1] + tx
    y_mapped = b * points[:, 0] + a * points[:, 1] + ty
    return np.stack([x_mapped, y_mapped], axis=1)


def ransac_similarity(src_points, dst_points, thresh=3.0, iters=200, rng=None):
    """
    RANSAC around the similarity fit so bad matches (mostly the moving person)
    don't skew it. Two pairs pin down a similarity; keep the largest inlier set
    and refit on it. Returns (params, inlier_mask).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    num_points = len(src_points)
    if num_points < 2:
        return None, None

    best_inlier_mask, best_inlier_count = None, -1
    all_indices = np.arange(num_points)
    for _ in range(iters):
        sample_indices = rng.choice(all_indices, size=2, replace=False)
        candidate_params = fit_similarity_lsq(src_points[sample_indices], dst_points[sample_indices])
        if candidate_params is None:
            continue
        residuals = np.linalg.norm(_apply_similarity(candidate_params, src_points) - dst_points, axis=1)
        inlier_mask = residuals < thresh
        inlier_count = int(inlier_mask.sum())
        if inlier_count > best_inlier_count:
            best_inlier_count, best_inlier_mask = inlier_count, inlier_mask

    if best_inlier_mask is None or best_inlier_count < 2:
        return fit_similarity_lsq(src_points, dst_points), np.ones(num_points, bool)  # nothing agreed, fit all
    return fit_similarity_lsq(src_points[best_inlier_mask], dst_points[best_inlier_mask]), best_inlier_mask


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
    # detect corners in the previous frame
    prev_corners = cv2.goodFeaturesToTrack(prev_gray, **_FEATURE_PARAMS)
    if prev_corners is None or len(prev_corners) < 8:
        return None

    tracked_corners, status_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_corners, None, **_LK_PARAMS)
    # forward-backward check: track back and keep only points that return home
    reprojected_corners, status_bwd, _ = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, tracked_corners, None, **_LK_PARAMS)
    fb_error = np.linalg.norm(prev_corners - reprojected_corners, axis=2).reshape(-1)
    good_mask = (status_fwd.reshape(-1) == 1) & (status_bwd.reshape(-1) == 1) & (fb_error < 1.0)
    if good_mask.sum() < 8:
        good_mask = (status_fwd.reshape(-1) == 1)  # too strict, fall back to forward tracks

    src_corners = prev_corners.reshape(-1, 2)[good_mask]
    dst_corners = tracked_corners.reshape(-1, 2)[good_mask]
    if len(src_corners) < 8:
        return None

    params, _ = ransac_similarity(src_corners, dst_corners, thresh=3.0, iters=200, rng=rng)
    if params is None:
        return None
    a, b, tx, ty = params
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
        for channel in range(trajectory.shape[1]):
            smoothed[:, channel] = gaussian_filter1d(trajectory[:, channel], sigma=sigma, mode='nearest')
    return smoothed


def _build_affine(dx, dy, d_angle):
    cos_a, sin_a = np.cos(d_angle), np.sin(d_angle)
    return np.array([[cos_a, -sin_a, dx], [sin_a, cos_a, dy]], dtype=np.float64)


def compute_stabilizing_transforms(input_path, smoothing_sigma=25.0, static_camera=True):
    """
    Pass 1: estimate per-frame motion, accumulate + smooth the path, and return one
    correction matrix per frame plus (n_frames, fps, w, h).
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rng = np.random.default_rng(0)  # fixed seed -> reproducible RANSAC
    ok, prev_frame = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read first frame of {input_path}")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    motions, last_motion = [], (0.0, 0.0, 0.0)
    while True:
        ok, cur_frame = cap.read()
        if not ok:
            break
        cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        motion = estimate_pair_motion(prev_gray, cur_gray, rng)  # motion = tx, ty, rotation
        if motion is None:
            motion = last_motion  # reuse last motion so the path stays continuous
        motions.append(motion)
        last_motion = motion
        prev_gray = cur_gray
    cap.release()

    num_frames = len(motions) + 1

    motions = np.array(motions, dtype=np.float64)
    trajectory = np.zeros((num_frames, 3), dtype=np.float64)   # absolute path, frame 0 at origin
    trajectory[1:] = np.cumsum(motions, axis=0)
    smoothed = smooth_trajectory(trajectory, sigma=smoothing_sigma, static_camera=static_camera)
    correction = smoothed - trajectory                       # per-frame shift onto the smooth path

    transforms = [_build_affine(correction[i, 0], correction[i, 1], correction[i, 2]) for i in range(num_frames)]
    return transforms, (num_frames, fps, width, height)


def warp_frames(input_path, transforms):
    """
    Pass 2 (generator): apply each correction, yielding (stabilized_bgr, valid_mask).
    valid_mask marks real pixels vs. the black border, for background subtraction.
    """
    cap = cv2.VideoCapture(input_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_mask = np.full((height, width), 255, np.uint8)
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        transform = transforms[frame_index]
        warped = cv2.warpAffine(frame, transform, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # nearest-neighbor keeps the mask a clean yes/no (no gray border bleed)
        valid = cv2.warpAffine(full_mask, transform, (width, height), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        yield warped, valid
        frame_index += 1
    cap.release()


def stabilize(input_path, output_path, smoothing_sigma=25.0, static_camera=True, fourcc='XVID'):
    """
    End to end: run both passes, write the stabilized video, return (transforms, meta).
    XVID so the .avi opens in Windows; OpenCV reads it back either way.
    """
    transforms, (num_frames, fps, width, height) = compute_stabilizing_transforms(
        input_path, smoothing_sigma, static_camera)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    for warped, _valid in warp_frames(input_path, transforms):
        writer.write(warped)
    writer.release()
    return transforms, (num_frames, fps, width, height)
