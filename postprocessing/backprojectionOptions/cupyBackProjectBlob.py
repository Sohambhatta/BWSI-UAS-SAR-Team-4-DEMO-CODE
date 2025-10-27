import pickle
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp   
from cupyx.scipy.ndimage import gaussian_filter
from skimage.feature import blob_log
import time
import sys
import Filter as filter
import os
from globalOptimization import SARSpatialOptimizer


startTime = time.time()


def retrieve_complex_data(real_data, range_bins):
    """
    Convert real data to complex data using a hilbert transform.
    Args:
        real_data: scan_data (np.ndarray): NxM array of real scan data
        range_bins: 1D numpy array of range bins.
    Returns:
        Numpy NxM array of complex scan data
    """
    #print(type(real_data), range_bins.shape[0])
    transform = cp.fft.fft(real_data, range_bins.shape[0])
    #print(transform.shape)
    midIndex = transform.shape[0] // 2
    H = cp.zeros(transform.shape[0])
    H[0] = 1
    H[midIndex] = 1
    H[1:midIndex] = 2
    transform *= H
    transform = cp.fft.ifft(transform, axis = -1)
    transform = cp.vstack(transform)
    c = 2.99792458e8  # Speed of light in m/s
    fc = 4.3e9
    return transform * cp.exp(-4j* cp.pi * range_bins[:, None] * fc/c)


#0: size 100, resolution 0.2, center (-100, -75) - Beaver
#21: size 20, resolution 0.04, center (25, -95) - Monopoly man
#18: size 4, resolution 0.01, center (-192.5, -160.8) - Linux penguin


BATCH_RUN = True # Run in batch mode, processing all files in the directory
NORMALIZE_DISTANCE = True # Remove points that are too close together based on a minimum distance threshold. If False, all points are used.
DYNAMIC_MIN_SAMPLING_DISTANCE = True # Uses med + 0.5*IQR distance as minimum sampling distance
SPACIAL_OPTIMIZATION = False # Use spatial binning to filter points based on distance instead of basic distance restriction. Much more computationally expensive, but sometimes more effective. 
MIN_SAMPLING_DISTANCE = 0.002 # meters, for nondynamic minimum sampling distance
NUM_LOOKS = 4 # number of looks, used to tune the spacial optimization. 

#TODO
SCALE_BY_DISTANCE = True # Increase DB by distance to center of imaging path
DB_INC_PER_METER = 0.4 # dB per meter from center of imaging path

EXPORT_PKL = False # Export the backprojected image as a pickle file for final submission

TERRAIN_SIZE = 3 # meters, size of the terrain in meters. The image will be TERRAIN_SIZE x TERRAIN_SIZE meters.
TERRAIN_RESOLUTION = 0.01 # meters, resolution of the terrain in meters. The image will have TERRAIN_SIZE / TERRAIN_RESOLUTION pixels in each dimension.

# Image center parameters
IMAGE_CENTER_X = 0.7 # meters 
IMAGE_CENTER_Y = -0.15 # meters

AUTO_DYNAMIC_RANGE = False # Automatically sets dynamic range based on lowest peak detected in the image. If False, uses NON_AUTO_DECREASE to set the dynamic range.
SHOW_DETECTED_PEAKS = False # Show detected peaks on the image. If False, no peaks are shown.
HILBERT = False # Use Hilbert transform to convert real data to complex data. If False, uses real data directly.

NON_AUTO_DECREASE = 5 # dB, how much to decrease the dynamic range from max if AUTO_DYNAMIC_RANGE is False

import os
if len(sys.argv) > 1:
    path = sys.argv[1]
    DIRECTORY = path
else:
    print("Error: Please provide a file or directory path as an argument.") 
    #DIRECTORY = 'DATAS//July29Flight6/BATCH//+0ms.pkl'
    #DIRECTORY = "DATAS//July29Flight2BII10//BATCH//+60ms.pkl"
    #DIRECTORY = 'DATAS/July31Flight2/BATCH/+0ms.pkl'
    #DIRECTORY = "DATAS/July31Flight4"
    #DIRECTORY = 'DATAS/July31Flight4/BATCH/-40ms.pkl'

LOOK_MODE = 1 # 0 for initial look, 1 for fine tuning, 2 for final look. 

# Quick set of parameters based on LOOK_MODE
match LOOK_MODE:
    case 0:
        AUTO_DYNAMIC_RANGE = True
        HILBERT = False
        DIRECTORY = "DATAS/July31Flight4"
        BATCH_RUN = True
        EXPORT_PKL = False
        TERRAIN_RESOLUTION = TERRAIN_RESOLUTION * 1.5
    case 1:
        AUTO_DYNAMIC_RANGE = False
        HILBERT = True
        DIRECTORY = 'DATAS/July31Flight4/BATCH/-40ms.pkl'
        BATCH_RUN = False
        EXPORT_PKL = False
    case 2:
        AUTO_DYNAMIC_RANGE = False
        HILBERT = True
        DIRECTORY = "DATAS/Aug1Flight2/+0ms.pkl"
        BATCH_RUN = False
        EXPORT_PKL = True


def open_pickle_file(filename):
    """Open a pickle file and return the loaded object.
    Args: 
        filename (str): Path to the pickle file.
    Returns: 
        Loaded object from the pickle file, or None if an error occurs.
    """
    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist.")
        return None
    except pickle.UnpicklingError:
        print("Error: The file could not be unpickled.")
        return None


def index_to_world(x_idx, y_idx):
    """Convert pixel indices to world coordinates.
    Args:
        x_idx (int): X index of the pixel.  
        y_idx (int): Y index of the pixel.
    Returns:
        Tuple of (x_world, y_world) in meters.
    """
    origin_x_world = IMAGE_CENTER_X - TERRAIN_SIZE  / 2
    origin_y_world = IMAGE_CENTER_Y - TERRAIN_SIZE / 2

    x_world = origin_x_world + x_idx * TERRAIN_RESOLUTION
    y_world = origin_y_world + y_idx * TERRAIN_RESOLUTION

    return x_world, y_world

def world_to_index(x_world, y_world): 
    """Convert world coordinates to pixel indices.
    Args:
        x_world (float): X coordinate in meters.
        y_world (float): Y coordinate in meters.
    Returns:
        Tuple of (x_idx, y_idx) as pixel indices.
    """
    origin_x_world = IMAGE_CENTER_X - TERRAIN_SIZE  / 2
    origin_y_world = IMAGE_CENTER_Y - TERRAIN_SIZE / 2

    x_idx = (x_world - origin_x_world) / TERRAIN_RESOLUTION
    y_idx = (y_world - origin_y_world) / TERRAIN_RESOLUTION

    return int(round(x_idx)), int(round(y_idx))

def oversample_filter(platform_pos, scan_data, min_distance):
    """Removes platform_pos and scan_data elements that are less than min_distance from the previous accepted location.
    Args: 
            platform_pos (np.ndarray): Nx3 array of platform positions in [x, y, z] format.
            scan_data (np.ndarray): NxM array of scan data corresponding to each platform position.
            min_distance (float): Minimum distance in meters between consecutive accepted positions, if DYNAMIC_MIN_SAMPLING_DISTANCE is False.
    Returns: Tuple of filtered platform positions and scan data.
    """
    # Calculate all pairwise distances between consecutive positions
    diffs = np.linalg.norm(np.diff(platform_pos, axis=0), axis=1)
    if len(diffs) > 0:
        min_dist = np.min(diffs)
        q1 = np.percentile(diffs, 25)
        median = np.median(diffs)
        q3 = np.percentile(diffs, 75)
        max_dist = np.max(diffs)
        iqr = q3 - q1
        print(f"Pre-filtering distances between positions:")
        print(f"Min: {min_dist:.4f} m, Q1: {q1:.4f} m, Median: {median:.4f} m, Q3: {q3:.4f} m, Max: {max_dist:.4f} m")
    else:
        print("Not enough positions to calculate statistics.")

    md = min_distance
    if DYNAMIC_MIN_SAMPLING_DISTANCE:
        if len(diffs) > 0:
            md = median + 0.5 * iqr
            print(f"Using median + 0.5 * iqr distance {md:.4f} m as minimum sampling distance.")
        else:
            print("Not enough positions to determine Q3 distance. Using static minimum distance.")
    filtered_pos = []
    filtered_data = []
    last_pos = None
    spacialOptimizer = SARSpatialOptimizer(platform_pos, scan_data)
    if not SPACIAL_OPTIMIZATION:
        filtered_pos, filtered_data = spacialOptimizer.distance_based_filter(
            min_distance=md,
            selection_method='sequential'
        )
    else:
        md /= max(1, NUM_LOOKS / 2)  
        filtered_pos, filtered_data = spacialOptimizer.spatial_binning_filter(
            bin_size = md,
            selection_method='center'
        )
    print(f"Filtered {len(platform_pos) - len(filtered_pos)} points from {len(platform_pos)} total points based on minimum distance of {md} meters.")
    return np.array(filtered_pos), np.array(filtered_data)

def run_backprojection(pkl_path):
    """Run backprojection algorithm on the provided pickle file.
    Args:
        pkl_path (str): Path to the pickle file containing scan records.
    Returns:
        Tuple of backprojection image, dynamic range minimum, dynamic range maximum, and lists of detected blob coordinates in world coordinates.
    """

    scan_records = open_pickle_file(pkl_path)
    if scan_records is None:
        print(f"No scan records loaded from {pkl_path}. Skipping.")
        return None, None, None, None, None
    global IMAGE_CENTER_X, IMAGE_CENTER_Y
    # Oversampling filter if NORMALIZE_DISTANCE is True
    platform_pos = scan_records['platform_pos']
    scan_data = scan_records['scan_data']
    if NORMALIZE_DISTANCE:
        platform_pos, scan_data = oversample_filter(platform_pos, scan_data, MIN_SAMPLING_DISTANCE)
    x = cp.arange(IMAGE_CENTER_X - TERRAIN_SIZE / 2, IMAGE_CENTER_X + TERRAIN_SIZE / 2, TERRAIN_RESOLUTION)
    y = cp.arange(IMAGE_CENTER_Y - TERRAIN_SIZE / 2, IMAGE_CENTER_Y + TERRAIN_SIZE / 2, TERRAIN_RESOLUTION)
    X, Y = cp.meshgrid(x, y)
    Z = cp.zeros_like(X)
    num_scans = len(scan_data)
    backproj_img_gpu = cp.zeros_like(X, dtype=cp.complex128)
    pixel_pos_gpu = cp.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    num_pixels = pixel_pos_gpu.shape[0]
    radar_pos_arr = cp.asarray(platform_pos)
    amplitudes_arr = cp.asarray(scan_data)
    range_bins_arr = cp.asarray(scan_records['range_bins'])
    print("processing")
    for scan_idx_gpu in range(num_scans):
        if scan_idx_gpu % max(1, num_scans // 10) == 0:
            percent = int(100 * scan_idx_gpu / num_scans)
            print(f"Progress: {percent}% ({scan_idx_gpu}/{num_scans})")
        radar_pos_gpu = radar_pos_arr[scan_idx_gpu]
        range_bin_gpu = range_bins_arr
        dists = cp.linalg.norm(pixel_pos_gpu - radar_pos_gpu, axis=1)
        if HILBERT:
            amp = retrieve_complex_data(amplitudes_arr[scan_idx_gpu], range_bin_gpu)
            amp = amp.T[0]
            amp_real = cp.interp(dists, range_bin_gpu, amp.real)
            amp_imag = cp.interp(dists, range_bin_gpu, amp.imag)
            interp_amp = amp_real + 1j * amp_imag
            interp_amp *= cp.exp(4j * cp.pi * dists * 4.3e9 / 2.99792458e8)
        else:
            amp_real = cp.interp(dists, range_bin_gpu, amplitudes_arr[scan_idx_gpu].real)
            amp_imag = cp.interp(dists, range_bin_gpu, amplitudes_arr[scan_idx_gpu].imag)
            interp_amp = amp_real + 1j * amp_imag
        valid = (dists >= range_bin_gpu[0]) & (dists <= range_bin_gpu[-1])
        pixel_sum = cp.zeros(num_pixels, dtype=cp.complex128)
        pixel_sum[valid] = interp_amp[valid]
        backproj_img_gpu = backproj_img_gpu.reshape(-1)
        backproj_img_gpu += pixel_sum
    backproj_img_gpu = backproj_img_gpu.reshape(X.shape)
    backproj_img_gpu = cp.abs(backproj_img_gpu)
    backproj_img_gpu_db = cp.zeros_like(backproj_img_gpu)
    nonzero_mask_gpu = backproj_img_gpu > 0
    backproj_img_gpu_db[nonzero_mask_gpu] = 20 * cp.log10(backproj_img_gpu[nonzero_mask_gpu])
    #smoothed_gpu_db = gaussian_filter(backproj_img_gpu_db, sigma=1.5)
    backproj_img_db = cp.asnumpy(backproj_img_gpu_db)
    #backproj_img_db = backproj_img_db.T

    if SCALE_BY_DISTANCE:
        # Compute the center of the imaging path
        start_x = scan_records['platform_pos'][:, 0].min()
        end_x = scan_records['platform_pos'][:, 0].max()
        start_y = scan_records['platform_pos'][:, 1].min()
        end_y = scan_records['platform_pos'][:, 1].max()
        start_z = scan_records['platform_pos'][:, 2].min()
        end_z = scan_records['platform_pos'][:, 2].max()

        center_x = (start_x + end_x) / 2
        center_y = (start_y + end_y) / 2
        center_z = (start_z + end_z) / 2

        # Prepare meshgrid for pixel coordinates (in world units)
        x_coords = np.arange(IMAGE_CENTER_X - TERRAIN_SIZE / 2, IMAGE_CENTER_X + TERRAIN_SIZE / 2, TERRAIN_RESOLUTION)
        y_coords = np.arange(IMAGE_CENTER_Y - TERRAIN_SIZE / 2, IMAGE_CENTER_Y + TERRAIN_SIZE / 2, TERRAIN_RESOLUTION)
        X, Y = np.meshgrid(x_coords, y_coords)
        Z = np.zeros_like(X)  # Assuming Z=0 for all pixels

        # Compute distance from each pixel to the center of the imaging path
        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2 + (Z - center_z) ** 2)

        # Increase amplitude by DB_INC_PER_METER for every squared meter from the center
        backproj_img_db = backproj_img_db + DB_INC_PER_METER * dist * dist

    
    print("Backprojection completed. Detecting peaks...")
    blobs = blob_log(backproj_img_db, min_sigma=4.2, max_sigma=10, threshold=1.3, log_scale=True, threshold_rel=0.65)
    db_max = np.max(backproj_img_db)
    filtered_blobs = []
    for y, x, sigma in blobs:
        x_idx = int(round(x))
        y_idx = int(round(y))
        x_min = max(0, x_idx - 3)
        x_max = min(backproj_img_db.shape[1], x_idx + 4)
        y_min = max(0, y_idx - 3)
        y_max = min(backproj_img_db.shape[0], y_idx + 4)
        local_max = np.max(backproj_img_db[y_min:y_max, x_min:x_max])
        if local_max > 0.9 * db_max:
            filtered_blobs.append((y, x, sigma, local_max))
    min_peak = np.inf
    for y, x, sigma, local_max in filtered_blobs:
        if local_max < min_peak:
            min_peak = local_max
            print(f"New minimum peak found: {local_max:.2f} db")
        print(f"Detected blob at ({x:.1f}, {y:.1f}) with scale {sigma:.2f}")
    if min_peak == np.inf:
        print("No valid peaks detected.")
        min_peak = 10
    vmin = min_peak - 1
    vmax = db_max - 1
    if not AUTO_DYNAMIC_RANGE:
        vmin = db_max - NON_AUTO_DECREASE
        vmax = db_max - 1
    blob_x_world = []
    blob_y_world = []
    for y, x, sigma, local_max in filtered_blobs:
        xw, yw = index_to_world(x, y)
        blob_x_world.append(xw)
        blob_y_world.append(yw)
        

    return backproj_img_db.T, vmin, vmax, blob_x_world, blob_y_world

if BATCH_RUN:
    batch_dir = os.path.join(DIRECTORY, "BATCH")
    if not os.path.exists(batch_dir):
        print(f"Batch directory {batch_dir} does not exist.")
    else:
        batch_files = [fname for fname in os.listdir(batch_dir) if fname.endswith('.pkl')]
        n = len(batch_files)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        axes = axes.flatten()
        for ax, fname in zip(axes, batch_files):
            pkl_path = os.path.join(batch_dir, fname)
            print(f"Running backprojection for {fname}")
            img, vmin, vmax, blob_x_world, blob_y_world = run_backprojection(pkl_path)
            if img is not None:
                ax.imshow(
                    img,
                    extent=[IMAGE_CENTER_Y - TERRAIN_SIZE / 2, IMAGE_CENTER_Y + TERRAIN_SIZE / 2, IMAGE_CENTER_X - TERRAIN_SIZE / 2, IMAGE_CENTER_X + TERRAIN_SIZE / 2],
                    origin='lower',
                    cmap='viridis',
                    vmin=vmin, vmax=vmax
                )
                ax.set_title(fname)
                ax.set_xlabel('Z (m)')
                ax.set_ylabel('X (m)')
                ax.grid(True, linestyle='--', alpha=0.5)
                if SHOW_DETECTED_PEAKS:
                    ax.scatter(blob_y_world,blob_x_world,  color='red', marker='x', label='Blobs')
                    ax.legend()
        plt.tight_layout()
        plt.show()
else:
    pkl_path = DIRECTORY
    img, vmin, vmax, blob_x_world, blob_y_world = run_backprojection(pkl_path)
    #img = filter.CLAHE(img, 3, 4.0)
    if EXPORT_PKL:
        print("Saving as pickle file for final submission...")
        x_axis = np.arange(IMAGE_CENTER_Y - TERRAIN_SIZE / 2, IMAGE_CENTER_Y + TERRAIN_SIZE / 2, TERRAIN_RESOLUTION)
        y_axis = np.arange(IMAGE_CENTER_X - TERRAIN_SIZE / 2, IMAGE_CENTER_X + TERRAIN_SIZE / 2, TERRAIN_RESOLUTION)
        final_dict = {
            "backprojected_img": img,
            "x_axis": x_axis,
            "z_axis": y_axis
        }
        with open("Team4_FinalSubmission.pkl", "wb") as f:
            pickle.dump(final_dict, f)
    plt.figure(figsize=(8, 6))
    plt.imshow(
        img,
        extent=[IMAGE_CENTER_Y - TERRAIN_SIZE / 2, IMAGE_CENTER_Y + TERRAIN_SIZE / 2, IMAGE_CENTER_X - TERRAIN_SIZE / 2, IMAGE_CENTER_X + TERRAIN_SIZE / 2],
        origin='lower',
        cmap='viridis',
        vmin=vmin, vmax=vmax
    )
    plt.colorbar(label='Amplitude (dB)')        
    plt.title("aligned_data.pkl")
    plt.xlabel('Z (m)')
    plt.ylabel('X (m)')
    plt.grid(True, linestyle='--', alpha=0.5)
    if SHOW_DETECTED_PEAKS:
        plt.scatter(blob_y_world,blob_x_world, color='red', marker='x', label='Blobs')
        plt.legend()
    plt.show()
