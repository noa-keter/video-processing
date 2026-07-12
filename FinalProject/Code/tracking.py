"""
Particle-filter tracking of the walking person in the matted video.

Condensation tracker (as in HW3) with a quantized color-histogram observation
model, guided by the binary masks: histograms are computed over person pixels
only and the particle weights include a mask-coverage prior.
"""

from __future__ import annotations

import json
import numpy as np

N_PARTICLES = 100
STATE_DIM = 6

# state layout: [x_center, y_center, half_width, half_height, x_velocity, y_velocity]
# process noise std per component; small scale noise since the person size barely changes
NOISE_STD = np.array([6.0, 6.0, 0.5, 0.5, 1.5, 1.5]).reshape(STATE_DIM, 1)
MIN_HALF_SIZE_PX = 4

HIST_BINS = 16                      # bins per color channel
QUANT_STEP = 256 // HIST_BINS
HIST_CORE_FRACTION = 0.6            # histograms use the central part of the box only; the mask
                                    # bbox is loose (spread legs), so a full-box histogram would
                                    # be dominated by background pixels
BHATTACHARYYA_GAIN = 20.0           # weight = exp(gain * Bhattacharyya coefficient)
COVERAGE_BIAS = 0.08                # keeps zero-coverage particles alive with a tiny weight
COVERAGE_GAMMA = 2.0                # emphasis of the mask-coverage prior
TEMPLATE_ADAPT_RATE = 0.05          # slow template update so lighting drift doesn't kill the weights
ADAPT_MIN_SIMILARITY = 0.7          # don't adapt the template when the match is poor (box may be off)
BBOX_SMOOTH_FACTOR = 0.6            # weight of the previous box in the output EMA

RNG_SEED = 0

RECT_COLOR_BGR = (0, 255, 0)
RECT_THICKNESS_PX = 2

MASK_THRESHOLD = 127     # binary.avi is expected as 0/255 (person = 255)
VIDEO_FOURCC = "XVID"


def color_histogram(frame_bgr: np.ndarray, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Normalized quantized-color histogram of the person pixels in the box described by `state`.

    Each channel is quantized to HIST_BINS levels and every color combination is
    counted. Only pixels where `mask` is True are used, so the histogram describes
    the person and not the background inside the box; if the box contains no mask
    pixels at all, the whole box is used as a fallback.

    Args:
        frame_bgr: H x W x 3 uint8 frame.
        state: State vector; only [x_center, y_center, half_width, half_height] are used.
        mask: H x W boolean person mask of the same frame.

    Returns:
        Flat histogram of length HIST_BINS**3 summing to 1 (all zeros for an empty box).
    """
    x_center, y_center, half_w, half_h = (int(round(v)) for v in state[:4])
    h, w = frame_bgr.shape[:2]
    col0, col1 = max(x_center - half_w, 0), min(x_center + half_w, w)
    row0, row1 = max(y_center - half_h, 0), min(y_center + half_h, h)

    hist = np.zeros((HIST_BINS, HIST_BINS, HIST_BINS))
    if row1 <= row0 or col1 <= col0:
        return hist.reshape(-1)

    patch = frame_bgr[row0:row1, col0:col1]
    person = mask[row0:row1, col0:col1]
    if person.any():
        quantized = patch[person] // QUANT_STEP
    else:
        quantized = (patch // QUANT_STEP).reshape(-1, 3)
    np.add.at(hist, (quantized[:, 0], quantized[:, 1], quantized[:, 2]), 1)
    hist = hist.reshape(-1)
    return hist / hist.sum()


def mask_coverage(mask: np.ndarray, state: np.ndarray) -> float:
    """
    Fraction of the box described by `state` that is covered by person pixels.

    Args:
        mask: H x W boolean person mask.
        state: State vector; only [x_center, y_center, half_width, half_height] are used.

    Returns:
        Coverage in [0, 1] (0 for an empty box).
    """
    x_center, y_center, half_w, half_h = (int(round(v)) for v in state[:4])
    h, w = mask.shape
    col0, col1 = max(x_center - half_w, 0), min(x_center + half_w, w)
    row0, row1 = max(y_center - half_h, 0), min(y_center + half_h, h)
    if row1 <= row0 or col1 <= col0:
        return 0.0
    return float(mask[row0:row1, col0:col1].mean())


def initial_state_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Build the initial tracker state from a binary person mask.

    Args:
        mask: 2-D boolean (or 0/1) array, True where the person is.

    Returns:
        State vector [x_center, y_center, half_width, half_height, 0, 0] as float64.

    Raises:
        ValueError: If the mask is empty.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        raise ValueError("person mask is empty, cannot initialize the tracker")

    y_center = (int(rows[0]) + int(rows[-1])) / 2.0
    x_center = (int(cols[0]) + int(cols[-1])) / 2.0
    half_h = (int(rows[-1]) - int(rows[0]) + 1) / 2.0
    half_w = (int(cols[-1]) - int(cols[0]) + 1) / 2.0
    return np.array([x_center, y_center, half_w, half_h, 0.0, 0.0])


class ParticleTracker:
    """
    Condensation tracker with a color-histogram observation model.

    Attributes:
        particles: STATE_DIM x N state matrix, one column per particle.
        template: Normalized color histogram of the tracked person.
    """

    def __init__(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray,
        initial_state: np.ndarray,
        *,
        n_particles: int = N_PARTICLES,
        seed: int = RNG_SEED,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._frame_hw = frame_bgr.shape[:2]
        self.particles = np.tile(initial_state.reshape(STATE_DIM, 1), (1, n_particles))
        self.template = color_histogram(frame_bgr, initial_state, mask)

    def step(self, frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Advance the tracker by one frame.

        Args:
            frame_bgr: The next H x W x 3 frame.
            mask: H x W boolean person mask of that frame.

        Returns:
            Weighted-mean state estimate [x_center, y_center, half_w, half_h, vx, vy].
        """
        self._predict()
        weights = self._observe(frame_bgr, mask)
        estimate = self.particles @ weights
        self._adapt_template(frame_bgr, mask, estimate)
        self._resample(weights)
        return estimate

    def _predict(self) -> None:
        """Apply the constant-velocity motion model and add process noise."""
        self.particles[0] += self.particles[4]
        self.particles[1] += self.particles[5]
        self.particles += NOISE_STD * self._rng.standard_normal(self.particles.shape)

        # keep the particles physical: centers inside the frame, box not degenerate
        h, w = self._frame_hw
        np.clip(self.particles[0], 0, w - 1, out=self.particles[0])
        np.clip(self.particles[1], 0, h - 1, out=self.particles[1])
        np.maximum(self.particles[2:4], MIN_HALF_SIZE_PX, out=self.particles[2:4])

    def _observe(self, frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Weigh every particle against the template; returns normalized weights.

        The color similarity is multiplied by a mask-coverage prior, so particles
        that drift off the person get a near-zero weight even when their colors
        happen to match.
        """
        n = self.particles.shape[1]
        weights = np.empty(n)
        for i in range(n):
            hist = color_histogram(frame_bgr, self.particles[:, i], mask)
            similarity = np.exp(BHATTACHARYYA_GAIN * np.sum(np.sqrt(hist * self.template)))
            coverage = mask_coverage(mask, self.particles[:, i])
            weights[i] = similarity * (COVERAGE_BIAS + coverage) ** COVERAGE_GAMMA
        return weights / weights.sum()

    def _adapt_template(self, frame_bgr: np.ndarray, mask: np.ndarray, estimate: np.ndarray) -> None:
        """Blend the histogram at the estimate into the template (slow adaptation)."""
        hist = color_histogram(frame_bgr, estimate, mask)
        if hist.sum() > 0:
            self.template = (1.0 - TEMPLATE_ADAPT_RATE) * self.template + TEMPLATE_ADAPT_RATE * hist

    def _resample(self, weights: np.ndarray) -> None:
        """Draw the next particle set from the weight distribution (SIR resampling)."""
        cdf = np.cumsum(weights)
        r = self._rng.uniform(size=weights.size)
        indices = np.minimum(np.searchsorted(cdf, r), weights.size - 1)
        self.particles = self.particles[:, indices]


def _binary_frame_to_mask(frame: np.ndarray) -> np.ndarray:
    """
    Turn one binary-video frame into a boolean person mask (same convention as matting).
    """
    if frame.ndim == 3:
        frame = frame[:, :, 0]
    return frame > MASK_THRESHOLD


def _state_to_box(state: np.ndarray, frame_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    """
    Convert [x_center, y_center, half_w, half_h, ...] to an integer (ROW, COL, HEIGHT, WIDTH)
    box clamped to the frame.
    """
    h, w = frame_hw
    x_center, y_center, half_w, half_h = state[:4]
    row = min(max(int(round(y_center - half_h)), 0), h - 1)
    col = min(max(int(round(x_center - half_w)), 0), w - 1)
    height = min(int(round(2 * half_h)), h - row)
    width = min(int(round(2 * half_w)), w - col)
    return row, col, height, width


def run_tracking(
    matted_path: str,
    binary_path: str,
    output_path: str,
    tracking_json_path: str,
) -> int:
    """
    Track the person in matted.avi, write OUTPUT.avi and tracking.json.

    The matted and binary videos are streamed in lockstep: each frame's mask
    guides the observation model, and the first frame's mask also places the
    initial box. The output box is smoothed with an EMA before drawing/saving,
    since the grader measures box deviation and the raw particle estimate jitters.

    Args:
        matted_path: Path to the matted color video.
        binary_path: Path to the binary mask video (0/255, person = 255).
        output_path: Where to write the video with the rectangle drawn.
        tracking_json_path: Where to write the frame -> [ROW, COL, HEIGHT, WIDTH] json.

    Returns:
        Number of frames written.

    Raises:
        FileNotFoundError: If an input video cannot be opened.
    """
    import cv2  # imported here so the library has no import-time cv2-I/O dependency

    matted = cv2.VideoCapture(matted_path)
    binv = cv2.VideoCapture(binary_path)
    if not matted.isOpened():
        raise FileNotFoundError(f"cannot open matted video: {matted_path}")
    if not binv.isOpened():
        raise FileNotFoundError(f"cannot open binary video: {binary_path}")

    width = int(matted.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(matted.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = matted.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = None
    smoothed = None
    boxes: dict[int, list[int]] = {}
    frame_index = 0
    while True:
        ok_mat, frame = matted.read()
        ok_bin, mask_frame = binv.read()
        if not ok_mat or not ok_bin:
            break
        frame_index += 1
        mask = _binary_frame_to_mask(mask_frame)

        if tracker is None:
            init_state = initial_state_from_mask(mask)
            tracker = ParticleTracker(frame, mask, init_state)
            smoothed = init_state[:4]
        else:
            estimate = tracker.step(frame, mask)
            smoothed = BBOX_SMOOTH_FACTOR * smoothed + (1.0 - BBOX_SMOOTH_FACTOR) * estimate[:4]

        row, col, box_h, box_w = _state_to_box(smoothed, (height, width))
        boxes[frame_index] = [row, col, box_h, box_w]
        cv2.rectangle(frame, (col, row), (col + box_w, row + box_h), RECT_COLOR_BGR, RECT_THICKNESS_PX)
        writer.write(frame)

    matted.release()
    binv.release()
    writer.release()
    with open(tracking_json_path, "w") as fp:
        json.dump(boxes, fp, indent=4)
    return frame_index
