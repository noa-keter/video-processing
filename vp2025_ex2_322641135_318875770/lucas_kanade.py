import cv2
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import griddata


# FILL IN YOUR ID
ID1 = 322641135
ID2 = 318875770


PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

WINDOW_SIZE = 5


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.

    Args:
        capture: cv2.VideoCapture object.

    Returns:
        parameters: dict. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


def build_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    """Coverts image to a pyramid list of size num_levels.

    First, create a list with the original image in it. Then, iterate over the
    levels. In each level, convolve the PYRAMID_FILTER with the image from the
    previous level. Then, decimate the result using indexing: simply pick
    every second entry of the result.
    Hint: Use signal.convolve2d with boundary='symm' and mode='same'.

    Args:
        image: np.ndarray. Input image.
        num_levels: int. The number of blurring / decimation times.

    Returns:
        pyramid: list. A list of np.ndarray of images.

    Note that the list length should be num_levels + 1 as the in first entry of
    the pyramid is the original image.
    You are not allowed to use cv2 PyrDown here (or any other cv2 method).
    We use a slightly different decimation process from this function.
    """
    pyramid = [image.copy()]
    """INSERT YOUR CODE HERE."""
    for i in range(num_levels):
        # Convolve the PYRAMID_FILTER with the image from the previous level
        convolved = signal.convolve2d(pyramid[-1], PYRAMID_FILTER, boundary='symm', mode='same')
        # Decimate by picking every second entry
        decimated = convolved[::2, ::2]
        pyramid.append(decimated)
    
    return pyramid


def lucas_kanade_step(I1: np.ndarray,
                      I2: np.ndarray,
                      window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform one Lucas-Kanade Step.

    This method receives two images as inputs and a window_size. It
    calculates the per-pixel shift in the x-axis and y-axis. That is,
    it outputs two maps of the shape of the input images. The first map
    encodes the per-pixel optical flow parameters in the x-axis and the
    second in the y-axis.

    (1) Calculate Ix and Iy by convolving I2 with the appropriate filters (
    see the constants in the head of this file).
    (2) Calculate It from I1 and I2.
    (3) Calculate du and dv for each pixel:
      (3.1) Start from all-zeros du and dv (each one) of size I1.shape.
      (3.2) Loop over all pixels in the image (you can ignore boundary pixels up
      to ~window_size/2 pixels in each side of the image [top, bottom,
      left and right]).
      (3.3) For every pixel, pretend the pixel’s neighbors have the same (u,
      v). This means that for NxN window, we have N^2 equations per pixel.
      (3.4) Solve for (u, v) using Least-Squares solution. When the solution
      does not converge, keep this pixel's (u, v) as zero.
    For detailed Equations reference look at slides 4 & 5 in:
    http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one is of the shape of the
        original image. dv encodes the optical flow parameters in rows and du
        in columns.
    """
    """INSERT YOUR CODE HERE.
    Calculate du and dv correctly.
    """
    # Spatial gradients from the second frame
    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, boundary='symm', mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, boundary='symm', mode='same')
    # Temporal gradient between frames
    It = I2 - I1

    # Initialize flow increments
    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)

    half_window = window_size // 2
    height, width = I1.shape

    # Iterate over valid pixels excluding borders
    for row in range(half_window, height - half_window):
        for col in range(half_window, width - half_window):
            Ix_window = Ix[row - half_window:row + half_window + 1,
                           col - half_window:col + half_window + 1]
            Iy_window = Iy[row - half_window:row + half_window + 1,
                           col - half_window:col + half_window + 1]
            It_window = It[row - half_window:row + half_window + 1,
                           col - half_window:col + half_window + 1]

            # Build least-squares system A * [u, v] = b
            A = np.stack((Ix_window.flatten(), Iy_window.flatten()), axis=1)
            b = -It_window.flatten()

            # Solve for (u, v) in the local window
            flow, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

            # Store local flow at the current pixel
            du[row, col] = flow[0]
            dv[row, col] = flow[1]
            
    return du, dv


def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Warp image using the optical flow parameters in u and v.

    Note that this method needs to support the case where u and v shapes do
    not share the same shape as of the image. We will update u and v to the
    shape of the image. The way to do it, is to:
    (1) cv2.resize to resize the u and v to the shape of the image.
    (2) Then, normalize the shift values according to a factor. This factor
    is the ratio between the image dimension and the shift matrix (u or v)
    dimension (the factor for u should take into account the number of columns
    in u and the factor for v should take into account the number of rows in v).

    As for the warping, use `scipy.interpolate`'s `griddata` method. Define the
    grid-points using a flattened version of the `meshgrid` of 0:w-1 and 0:h-1.
    The values here are simply image.flattened().
    The points you wish to interpolate are, again, a flattened version of the
    `meshgrid` matrices - don't forget to add them v and u.
    Use `np.nan` as `griddata`'s fill_value.
    Finally, fill the nan holes with the source image values.
    Hint: For the final step, use np.isnan(image_warp).

    Args:
        image: np.ndarray. Image to warp.
        u: np.ndarray. Optical flow parameters corresponding to the columns.
        v: np.ndarray. Optical flow parameters corresponding to the rows.

    Returns:
        image_warp: np.ndarray. Warped image.
    """
    image_warp = image.copy()
    """INSERT YOUR CODE HERE.
    Replace image_warp with something else.
    """
    height, width = image.shape

    # Resize flow fields to image size and scale shifts accordingly
    if u.shape != image.shape:
        u_scale = width / u.shape[1]
        u = cv2.resize(u, (width, height)) * u_scale
    if v.shape != image.shape:
        v_scale = height / v.shape[0]
        v = cv2.resize(v, (width, height)) * v_scale

    # Build source grid points and values
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    points = np.column_stack((x_grid.flatten(), y_grid.flatten()))
    values = image.flatten()

    # Target points are shifted by (u, v)
    warp_points = (x_grid + u, y_grid + v)
    image_warp = griddata(points, values, warp_points, fill_value=np.nan)

    # Fill undefined pixels with original image values
    nan_mask = np.isnan(image_warp)
    image_warp[nan_mask] = image[nan_mask]
    return image_warp


def lucas_kanade_optical_flow(I1: np.ndarray,
                              I2: np.ndarray,
                              window_size: int,
                              max_iter: int,
                              num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the optical flow parameters in rows and u in
        columns.

    Recipe:
        (1) Since the image is going through a series of decimations,
        we would like to resize the image shape to:
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (2) Build pyramids for the two images.
        (3) Initialize u and v as all-zero matrices in the shape of I1.
        (4) For every level in the image pyramid (start from the smallest
        image):
          (4.1) Warp I2 from that level according to the current u and v.
          (4.2) Repeat for num_iterations:
            (4.2.1) Perform a Lucas Kanade Step with the I1 decimated image
            of the current pyramid level and the current I2_warp to get the
            new I2_warp.
          (4.3) For every level which is not the image's level, perform an
          image resize (using cv2.resize) to the next pyramid level resolution
          and scale u and v accordingly.
    """
    """INSERT YOUR CODE HERE.
        Replace image_warp with something else.
        """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                  h_factor * (2 ** (num_levels - 1 + 1)))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyarmid_I2 = build_pyramid(I2, num_levels)
    # start from u and v in the size of smallest image
    u = np.zeros(pyarmid_I2[-1].shape)
    v = np.zeros(pyarmid_I2[-1].shape)
    """INSERT YOUR CODE HERE.
       Replace u and v with their true value."""
    for level in range(num_levels, -1, -1):
        # Work from smallest pyramid level to full resolution
        I1_level = pyramid_I1[level]
        I2_level = pyarmid_I2[level]

        # Warp I2 at the current level using current flow
        I2_warp = warp_image(I2_level, u, v)

        # Refine flow by iterative L.K updates
        for i in range(max_iter):
            du, dv = lucas_kanade_step(I1_level, I2_warp, window_size)
            u += du
            v += dv
            I2_warp = warp_image(I2_level, u, v)

        # Upscale flow for the next pyramid level
        if level > 0:
            next_shape = pyarmid_I2[level - 1].shape
            u = cv2.resize(u, (next_shape[1], next_shape[0])) * 2
            v = cv2.resize(v, (next_shape[1], next_shape[0])) * 2
    if u.shape != I1.shape:
        target_height, target_width = I1.shape
        u_scale = target_width / u.shape[1]
        v_scale = target_height / v.shape[0]
        u = cv2.resize(u, (target_width, target_height)) * u_scale
        v = cv2.resize(v, (target_width, target_height)) * v_scale
    return u, v


def lucas_kanade_video_stabilization(input_video_path: str,
                                     output_video_path: str,
                                     window_size: int,
                                     max_iter: int,
                                     num_levels: int) -> None:
    """Use LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.

    Recipe:
        (1) Open a VideoCapture object of the input video and read its
        parameters.
        (2) Create an output video VideoCapture object with the same
        parameters as in (1) in the path given here as input.
        (3) Convert the first frame to grayscale and write it as-is to the
        output video.
        (4) Resize the first frame as in the Full-Lucas-Kanade function to
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (5) Create a u and a v which are og the size of the image.
        (6) Loop over the frames in the input video (use tqdm to monitor your
        progress) and:
          (6.1) Resize them to the shape in (4).
          (6.2) Feed them to the lucas_kanade_optical_flow with the previous
          frame.
          (6.3) Use the u and v maps obtained from (6.2) and compute their
          mean values over the region that the computation is valid (exclude
          half window borders from every side of the image).
          (6.4) Update u and v to their mean values inside the valid
          computation region.
          (6.5) Add the u and v shift from the previous frame diff such that
          frame in the t is normalized all the way back to the first frame.
          (6.6) Save the updated u and v for the next frame (so you can
          perform step 6.5 for the next frame.
          (6.7) Finally, warp the current frame with the u and v you have at
          hand.
          (6.8) We highly recommend you to save each frame to a directory for
          your own debug purposes. Erase that code when submitting the exercise.
       (7) Do not forget to gracefully close all VideoCapture and to destroy
       all windows.
    """
    """INSERT YOUR CODE HERE."""
    # Open the input video and read its parameters.
    input_cap = cv2.VideoCapture(input_video_path)
    video_params = get_video_parameters(input_cap)
    
    width = video_params['width']
    height = video_params['height']
    fps = video_params['fps']
    frame_count = video_params['frame_count']
    
    # Create the output writer using the same parameters.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Use isColor=False because we are processing and writing grayscale frames
    out_cap = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)
    
    # Convert the first frame to grayscale and write it directly.
    ret, first_frame = input_cap.read()
    if not ret:
        input_cap.release()
        out_cap.release()
        return

    prev_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    out_cap.write(prev_frame_gray)
    
    # Resize the first frame to a pyramid-friendly size.
    h_factor = int(np.ceil(height / (2 ** (num_levels - 1))))
    w_factor = int(np.ceil(width / (2 ** (num_levels - 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1)), 
                  h_factor * (2 ** (num_levels - 1)))
    
    prev_frame_resized = cv2.resize(prev_frame_gray, IMAGE_SIZE)
    
    # Initialize cumulative shifts in the original frame size.
    u = np.zeros((height, width), dtype=np.float32)
    v = np.zeros((height, width), dtype=np.float32)
    
    # Half-window size used to ignore boundary pixels.
    half_w = window_size // 2

    # Process the rest of the frames.
    for _ in tqdm(range(1, frame_count), desc="Stabilizing Video"):
        ret, frame = input_cap.read()
        if not ret:
            break
            
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize current frame to match the flow estimation size.
        curr_frame_resized = cv2.resize(curr_frame_gray, IMAGE_SIZE)
        
        # Estimate flow between consecutive resized frames.
        du, dv = lucas_kanade_optical_flow(prev_frame_resized, 
                                           curr_frame_resized, 
                                           window_size, 
                                           max_iter, 
                                           num_levels)
        
        # Compute mean flow over the valid region (exclude borders).
        if half_w > 0:
            valid_du = du[half_w:-half_w, half_w:-half_w]
            valid_dv = dv[half_w:-half_w, half_w:-half_w]
        else:
            valid_du = du
            valid_dv = dv
            
        mean_du = np.nanmean(valid_du)
        mean_dv = np.nanmean(valid_dv)
        
        # Accumulate mean shifts to align back to the first frame.
        u += mean_du
        v += mean_dv
        
        # Warp the current frame using the cumulative shifts.
        warped_frame = warp_image(curr_frame_gray, u, v)
        
        # Convert back to uint8 properly for VideoWriter
        warped_frame_uint8 = np.clip(warped_frame, 0, 255).astype(np.uint8)
        out_cap.write(warped_frame_uint8)
        
        # Keep the resized frame for the next iteration.
        prev_frame_resized = curr_frame_resized
        
    # Release resources.
    input_cap.release()
    out_cap.release()
    cv2.destroyAllWindows()


def faster_lucas_kanade_step(I1: np.ndarray,
                             I2: np.ndarray,
                             window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Faster implementation of a single Lucas-Kanade Step.

    (1) If the image is small enough (you need to design what is good
    enough), simply return the result of the good old lucas_kanade_step
    function.
    (2) Otherwise, find corners in I2 and calculate u and v only for these
    pixels.
    (3) Return maps of u and v which are all zeros except for the corner
    pixels you found in (2).

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one of the shape of the
        original image. dv encodes the shift in rows and du in columns.
    """

    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    """INSERT YOUR CODE HERE.
    Calculate du and dv correctly.
    """
    height, width = I1.shape
    if min(height, width) <= window_size * 5:
        return lucas_kanade_step(I1, I2, window_size)

    I2_float = I2.astype(np.float32)
    corner_response = cv2.cornerHarris(I2_float, 2, 3, 0.04)
    threshold = 0.01 * corner_response.max()
    corner_points = np.argwhere(corner_response > threshold)

    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, boundary='symm', mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, boundary='symm', mode='same')
    It = I2 - I1

    half_window = window_size // 2

    for row, col in corner_points:
        if (row - half_window < 0 or row + half_window >= height or
                col - half_window < 0 or col + half_window >= width):
            continue

        Ix_window = Ix[row - half_window:row + half_window + 1,
                       col - half_window:col + half_window + 1]
        Iy_window = Iy[row - half_window:row + half_window + 1,
                       col - half_window:col + half_window + 1]
        It_window = It[row - half_window:row + half_window + 1,
                       col - half_window:col + half_window + 1]

        A = np.stack((Ix_window.flatten(), Iy_window.flatten()), axis=1)
        b = -It_window.flatten()

        flow, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        du[row, col] = flow[0]
        dv[row, col] = flow[1]
    return du, dv


def faster_lucas_kanade_optical_flow(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels .

    Use faster_lucas_kanade_step instead of lucas_kanade_step.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the shift in rows and u in columns.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** num_levels)))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** num_levels)))
    IMAGE_SIZE = (w_factor * (2 ** num_levels),
                  h_factor * (2 ** num_levels))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    pyramid_I1 = build_pyramid(I1, num_levels)  # create levels list for I1
    pyarmid_I2 = build_pyramid(I2, num_levels)  # create levels list for I1
    u = np.zeros(pyarmid_I2[-1].shape)  # create u in the size of smallest image
    v = np.zeros(pyarmid_I2[-1].shape)  # create v in the size of smallest image
    """INSERT YOUR CODE HERE.
    Replace u and v with their true value."""
    for level in range(num_levels, -1, -1):
        I1_level = pyramid_I1[level]
        I2_level = pyarmid_I2[level]

        I2_warp = warp_image(I2_level, u, v)

        for i in range(max_iter):
            du, dv = faster_lucas_kanade_step(I1_level, I2_warp, window_size)
            u += du
            v += dv
            I2_warp = warp_image(I2_level, u, v)

        if level > 0:
            next_shape = pyarmid_I2[level - 1].shape
            u = cv2.resize(u, (next_shape[1], next_shape[0])) * 2
            v = cv2.resize(v, (next_shape[1], next_shape[0])) * 2

    if u.shape != I1.shape:
        target_height, target_width = I1.shape
        u_scale = target_width / u.shape[1]
        v_scale = target_height / v.shape[0]
        u = cv2.resize(u, (target_width, target_height)) * u_scale
        v = cv2.resize(v, (target_width, target_height)) * v_scale
    return u, v


def lucas_kanade_faster_video_stabilization(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.
    """
    """INSERT YOUR CODE HERE."""
   # Open input video
    capture = cv2.VideoCapture(input_video_path)
    if not capture.isOpened():
        return

    # Prepare output video writer using input parameters
    params = get_video_parameters(capture)
    fourcc = cv2.VideoWriter_fourcc(*"XVID") # [cite: 204]
    writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        params["fps"],
        (params["width"], params["height"]),
        isColor=False,
    )

    # Read first frame and write it as-is
    ret, prev_frame = capture.read()
    if not ret:
        capture.release()
        writer.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    writer.write(prev_gray)

    # Resize frames to pyramid-friendly size
    h_factor = int(np.ceil(params["height"] / (2 ** (num_levels - 1))))
    w_factor = int(np.ceil(params["width"] / (2 ** (num_levels - 1))))
    image_size = (w_factor * (2 ** (num_levels - 1)),
                  h_factor * (2 ** (num_levels - 1)))

    prev_gray_resized = cv2.resize(prev_gray, image_size)

    # Initialize cumulative shifts in the shape of the resized image
    u = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    v = np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    half_window = window_size // 2

    for _ in tqdm(range(1, params["frame_count"]), desc="Faster LK Stabilizing"):
        ret, frame = capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, image_size)

        # Estimate optical flow between consecutive frames using the faster method
        du, dv = faster_lucas_kanade_optical_flow(
            prev_gray_resized,
            gray_resized,
            window_size,
            max_iter,
            num_levels,
        )

        # Compute mean shift over valid region (exclude borders)
        if half_window > 0:
            valid_du = du[half_window:-half_window, half_window:-half_window]
            valid_dv = dv[half_window:-half_window, half_window:-half_window]
        else:
            valid_du = du
            valid_dv = dv

        # Isolate only the actual corner shifts
        non_zero_du = valid_du[valid_du != 0]
        non_zero_dv = valid_dv[valid_dv != 0]

        # Calculate mean safely
        du_mean = np.mean(non_zero_du) if len(non_zero_du) > 0 else 0.0
        dv_mean = np.mean(non_zero_dv) if len(non_zero_dv) > 0 else 0.0

        # Accumulate motion using NumPy broadcasting
        u += du_mean
        v += dv_mean

        # Warp the ORIGINAL high-res frame using the resized u and v matrices
        stabilized = warp_image(gray, u, v)
        
        # Clip and convert to correct dtype for OpenCV VideoWriter
        stabilized_uint8 = np.clip(stabilized, 0, 255).astype(np.uint8)
        writer.write(stabilized_uint8)

        # Update reference frame
        prev_gray_resized = gray_resized

    # Clean up resources
    capture.release()
    writer.release()
    cv2.destroyAllWindows()


def lucas_kanade_faster_video_stabilization_fix_effects(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int, start_rows: int = 10,
        start_cols: int = 2, end_rows: int = 30, end_cols: int = 30) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
        start_rows: int. The number of lines to cut from top.
        end_rows: int. The number of lines to cut from bottom.
        start_cols: int. The number of columns to cut from left.
        end_cols: int. The number of columns to cut from right.

    Returns:
        None.
    """
    """INSERT YOUR CODE HERE."""
    capture = cv2.VideoCapture(input_video_path)
    if not capture.isOpened():
        return

    params = get_video_parameters(capture)
    output_height = params["height"] - start_rows - end_rows
    output_width = params["width"] - start_cols - end_cols
    if output_height <= 0 or output_width <= 0:
        capture.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        params["fps"],
        (output_width, output_height),
        isColor=False,
    )

    ret, prev_frame = capture.read()
    if not ret:
        capture.release()
        writer.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cropped_prev = prev_gray[start_rows:params["height"] - end_rows,
                             start_cols:params["width"] - end_cols]
    writer.write(cropped_prev)

    h_factor = int(np.ceil(params["height"] / (2 ** (num_levels - 1))))
    w_factor = int(np.ceil(params["width"] / (2 ** (num_levels - 1))))
    image_size = (w_factor * (2 ** (num_levels - 1)),
                  h_factor * (2 ** (num_levels - 1)))

    prev_gray_resized = cv2.resize(prev_gray, image_size)

    u_cumulative = np.zeros(prev_gray_resized.shape)
    v_cumulative = np.zeros(prev_gray_resized.shape)
    total_u = 0.0
    total_v = 0.0

    half_window = window_size // 2

    for i in tqdm(
        range(1, params["frame_count"]),
        desc="Faster LK (no-borders)",
        leave=False,
        position=0,
        dynamic_ncols=False,
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        ret, frame = capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, image_size)

        du, dv = faster_lucas_kanade_optical_flow(
            prev_gray_resized,
            gray_resized,
            window_size,
            max_iter,
            num_levels,
        )

        if du.shape != u_cumulative.shape:
            target_height, target_width = u_cumulative.shape
            u_scale = target_width / du.shape[1]
            v_scale = target_height / du.shape[0]
            du = cv2.resize(du, (target_width, target_height)) * u_scale
            dv = cv2.resize(dv, (target_width, target_height)) * v_scale

        valid_rows = slice(half_window, -half_window or None)
        valid_cols = slice(half_window, -half_window or None)
        du_mean = np.mean(du[valid_rows, valid_cols])
        dv_mean = np.mean(dv[valid_rows, valid_cols])

        du = np.full_like(du, du_mean)
        dv = np.full_like(dv, dv_mean)

        total_u += du_mean
        total_v += dv_mean
        u_cumulative = np.full_like(u_cumulative, total_u)
        v_cumulative = np.full_like(v_cumulative, total_v)

        stabilized = warp_image(gray_resized, u_cumulative, v_cumulative)
        stabilized = cv2.resize(stabilized, (params["width"], params["height"]))
        cropped = stabilized[start_rows:params["height"] - end_rows,
                             start_cols:params["width"] - end_cols]
        writer.write(cropped.astype(prev_gray.dtype))

        prev_gray_resized = gray_resized

    capture.release()
    writer.release()
    cv2.destroyAllWindows()


