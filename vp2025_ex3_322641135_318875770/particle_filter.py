import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# change IDs to your IDs.
ID1 = "318875770"
ID2 = "322641135"

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# process noise std per state component [Xc, Yc, W/2, H/2, Xv, Yv]
NOISE_STD = np.array([8.0, 8.0, 0.0, 0.0, 2.0, 2.0]).reshape(6, 1)

# Initial Settings
s_initial = [297,    # x center
             139,    # y center
              16,    # half width
              43,    # half height
               0,    # velocity x
               0]    # velocity y


def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    s_prior = s_prior.astype(float)
    state_drifted = s_prior.copy()

    # constant velocity model: move center by its velocity
    state_drifted[0, :] = s_prior[0, :] + s_prior[4, :]
    state_drifted[1, :] = s_prior[1, :] + s_prior[5, :]

    # add white noise (keep width/height fixed)
    state_drifted = state_drifted + NOISE_STD * np.random.randn(*state_drifted.shape)

    state_drifted = state_drifted.astype(int)
    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    state = np.floor(state)
    state = state.astype(int)
    hist = np.zeros((16, 16, 16))

    x_center, y_center, half_width, half_height = state[0], state[1], state[2], state[3]

    # crop the box, clamped to the image (negative indices would wrap around)
    img_h, img_w = image.shape[0], image.shape[1]
    x_min, x_max = max(x_center - half_width, 0), min(x_center + half_width, img_w)
    y_min, y_max = max(y_center - half_height, 0), min(y_center + half_height, img_h)
    patch = image[y_min:y_max, x_min:x_max, :]

    # quantize each channel to 4 bits (0-15) and count every color combination
    quantized = (patch.astype(int) // 16).reshape(-1, 3)
    np.add.at(hist, (quantized[:, 0], quantized[:, 1], quantized[:, 2]), 1)

    hist = np.reshape(hist, 16 * 16 * 16)

    # normalize
    hist = hist/sum(hist)

    return hist


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """
    N = previous_state.shape[1]

    # Draw N uniform random numbers in [0, 1)
    r = np.random.uniform(0, 1, size=N)
    # For each random number, find the index of the first particle whose cdf is greater than or equal to it
    indices = np.searchsorted(cdf, r, side='left')
    # Guard against float rounding at cdf=1
    indices = np.minimum(indices, N - 1)
    # Sample the previous state using the indices to get the next state 
    S_next = previous_state[:, indices]

    return S_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """
    # weight from the Bhattacharyya coefficient: w = exp(20 * sum(sqrt(p*q)))
    distance = np.exp(20 * np.sum(np.sqrt(p * q)))
    return distance


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:
    fig, ax = plt.subplots(1)
    image = image[:,:,::-1]
    plt.imshow(image)
    plt.title(ID + "- Frame number = " + str(frame_index))

    # Avg particle box: weighted mean of all particles, then center -> top-left corner
    x_center = np.sum(W * state[0, :])
    y_center = np.sum(W * state[1, :])
    half_w = np.sum(W * state[2, :])
    half_h = np.sum(W * state[3, :])
    (x_avg, y_avg, w_avg, h_avg) = (x_center - half_w, y_center - half_h, 2 * half_w, 2 * half_h)


    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box: the single particle with the largest weight
    i_max = np.argmax(W)
    x_center, y_center, half_w, half_h = state[0, i_max], state[1, i_max], state[2, i_max], state[3, i_max]
    (x_max, y_max, w_max, h_max) = (x_center - half_w, y_center - half_h, 2 * half_w, 2 * half_h)

    rect = patches.Rectangle((x_max, y_max), w_max, h_max, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show(block=False)

    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state


def main():
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame)

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    W = np.zeros(N)
    for i in range(N):
        p = compute_normalized_histogram(image, S[:, i])
        W[i] = bhattacharyya_distance(p, q)
    W = W / np.sum(W)
    C = np.cumsum(W)

    images_processed = 1

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    for image_name in image_name_list[1:]:

        S_prev = S

        # LOAD NEW IMAGE FRAME
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        current_image = cv2.imread(image_path)

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        # score each predicted particle on the new frame against the fixed target q
        W = np.zeros(N)
        for i in range(N):
            p = compute_normalized_histogram(current_image, S[:, i])
            W[i] = bhattacharyya_distance(p, q)
        W = W / np.sum(W)
        C = np.cumsum(W)

        # CREATE DETECTOR PLOTS
        images_processed += 1
        if 0 == images_processed%10:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()
