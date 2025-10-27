import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_log
import time
import concurrent.futures
import os
import threading
import Filter

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
    transform = np.fft.fft(real_data, range_bins.shape[0])
    #print(transform.shape)
    midIndex = transform.shape[0] // 2
    H = np.zeros(transform.shape[0])
    H[0] = 1
    H[midIndex] = 1
    H[1:midIndex] = 2
    transform *= H
    transform = np.fft.ifft(transform, axis = -1)
    transform =np.vstack(transform)
    c = 2.99792458e8  # Speed of light in m/s
    fc = 4.3e9
    return transform * np.exp(-4j* np.pi * range_bins[:, None] * fc/c)

#0: size 100, resolution 0.2, center (-100, -75) (maybe?)
#21: size 20, resolution 0.04, center (25, -95)



#DIRECTORY = '5_point_scatter.pkl'
#DIRECTORY = 'marathon_21.pkl'
#DIRECTORY = 'SIM//aligned_data.pkl'
import sys
import os


# if len(sys.argv) > 1:
#     path = sys.argv[1]
#     if os.path.isdir(path):
#         DIRECTORY = os.path.join(path, 'aligned_data.pkl')
#     else:
#         DIRECTORY = path
# else:
#     print("Error: Please provide a file or directory path as an argument.")
#     sys.exit(1)


DIRECTORY = r"c:\Users\soham\Coding Projects\Team_Four_Repo-Simple\Aug1Flight1\aligned_data.pkl"
TERRAIN_SIZE = 3
TERRAIN_RESOLUTION = 0.01

HILBERT = True

# Image center parameters
IMAGE_CENTER_X = 0.7 # meters
IMAGE_CENTER_Y = -0.15 # meters

AUTO_DYNAMIC_RANGE = True
SHOW_DETECTED_PEAKS = False

x = np.arange(IMAGE_CENTER_X - TERRAIN_SIZE / 2, IMAGE_CENTER_X + TERRAIN_SIZE / 2, TERRAIN_RESOLUTION)
y = np.arange(IMAGE_CENTER_Y - TERRAIN_SIZE / 2, IMAGE_CENTER_Y + TERRAIN_SIZE / 2, TERRAIN_RESOLUTION)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

def open_pickle_file(filename):
    """Open a pickle file and return the loaded object."""
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
    
print("Loading scan records from pickle file...")

scan_records = open_pickle_file(DIRECTORY)

# Check for valid scan records 
if scan_records is None:
    print("No scan records loaded. Exiting.")
    exit()
else:
    print(scan_records["platform_pos"].shape)
    print(scan_records["scan_data"].shape)
    print(scan_records["range_bins"].shape)
    #print(scan_records.keys())
    #exit()
    pass

num_scans = len(scan_records['scan_data'])

# Init backprojection image
backproj_img = np.zeros_like(X)
pixel_pos = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
num_pixels = pixel_pos.shape[0]

# Prepare scan data arrays for GPU
radar_pos_arr = np.asarray(scan_records['platform_pos'])
amplitudes_arr = np.asarray(scan_records['scan_data'])
#amplitudes_arr = np.abs(amplitudes_arr)
#amplitudes_arr = amplitudes_arr.real  # Ensure we are working with real values
range_bins_arr = np.asarray(scan_records['range_bins'])

print("processing")

def process_chunk(chunk_indices):
    # chunk_indices: indices of the pixels this thread will process
    chunk_pixel_pos = pixel_pos[chunk_indices]
    chunk_sum = np.zeros(len(chunk_indices), dtype=np.complex128) 

    #for logging
    total_scans = num_scans
    progress_interval = max(1, total_scans // 10)
    thread_id = threading.get_ident()

    for scan_idx in range(num_scans):
        radar_pos = radar_pos_arr[scan_idx]
        range_bin = range_bins_arr
        dists = np.linalg.norm(chunk_pixel_pos - radar_pos, axis=1)
        # Interpolate real and imaginary parts separately
        if HILBERT:
            amp = retrieve_complex_data(amplitudes_arr[scan_idx], range_bin)
            amp = amp.T[0]

            amp_real = np.interp(dists, range_bin, amp.real)
            amp_imag = np.interp(dists, range_bin, amp.imag)
            interp_amp = amp_real + 1j * amp_imag

            #print(dists.shape, interp_amp.shape)
            
            interp_amp *= np.exp(4j * np.pi * dists * 4.3e9 / 2.99792458e8)  # Apply phase shift
        else:
            amp_real = np.interp(dists, range_bin, amplitudes_arr[scan_idx].real)
            amp_imag = np.interp(dists, range_bin, amplitudes_arr[scan_idx].imag)
            interp_amp = amp_real + 1j * amp_imag 
        # Mask for valid distances
        valid = (dists >= range_bin[0]) & (dists <= range_bin[-1])
        chunk_sum[valid] += interp_amp[valid]

        #progress check
        if (scan_idx + 1) % progress_interval == 0 or (scan_idx + 1) == total_scans:
            percent = int(100 * (scan_idx + 1) / total_scans)
            print(f"Thread {thread_id}: {percent}% of scans processed for its chunk")

    return chunk_indices, chunk_sum

# Split indices into N chunks
num_threads = os.cpu_count() - 4  # or use os.cpu_count()

print(f"{num_threads} threads processing")
indices = np.arange(num_pixels)
chunks = np.array_split(indices, num_threads)

results = np.zeros(num_pixels, dtype=np.complex128)
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
    for future in concurrent.futures.as_completed(futures):
        chunk_indices, chunk_sum = future.result()
        results[chunk_indices] = chunk_sum

backproj_img = results.reshape(X.shape)
        # Move result back to CPU and plot
backproj_img = np.abs(backproj_img)  # Ensure we are working with absolute values
backproj_img_db = np.zeros_like(backproj_img)
nonzero_mask = backproj_img > 0
backproj_img_db[nonzero_mask] = 20 * np.log10(backproj_img[nonzero_mask])

smoothed_db = gaussian_filter(backproj_img_db, sigma=1.5)

# Set dB value to 0 if raw value is 0, otherwise convert to dB


print("Backprojection completed. Detecting peaks...")

def index_to_world(x_idx, y_idx):
    origin_x_world = IMAGE_CENTER_X - TERRAIN_SIZE  / 2
    origin_y_world = IMAGE_CENTER_Y - TERRAIN_SIZE / 2

    x_world = origin_x_world + x_idx * TERRAIN_RESOLUTION
    y_world = origin_y_world + y_idx * TERRAIN_RESOLUTION

    return x_world, y_world

def world_to_index(x_world, y_world): 
    origin_x_world = IMAGE_CENTER_X - TERRAIN_SIZE  / 2
    origin_y_world = IMAGE_CENTER_Y - TERRAIN_SIZE / 2

    x_idx = (x_world - origin_x_world) / TERRAIN_RESOLUTION
    y_idx = (y_world - origin_y_world) / TERRAIN_RESOLUTION

    return int(round(x_idx)), int(round(y_idx))


blobs = blob_log(backproj_img_db, 
                 min_sigma=4.2, max_sigma=10, 
                 threshold=1.3, log_scale=True, threshold_rel=0.65
                 )

db_max = np.max(backproj_img_db)

# Filter blobs based on local maximum in 3-pixel radius
filtered_blobs = []
for y, x, sigma in blobs:
    x_idx = int(round(x))
    y_idx = int(round(y))
    # Define bounds for 3-pixel radius neighborhood
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

# Dynamic range calculation, to be improved
vmin = min_peak - 10
vmax = db_max - 1

# Convert blob indices to world coordinates for plotting
blob_x_world = []
blob_y_world = []
for y, x, sigma, local_max in filtered_blobs:
    xw, yw = index_to_world(x, y)
    blob_x_world.append(xw)
    blob_y_world.append(yw)


endTime = time.time()
print(f"Total processing time: {endTime - startTime:.2f} seconds")

# Plot the backprojected image with detected peaks
plt.figure(figsize=(8, 6))

filtered = smoothed_db

filtered = filtered.T

if AUTO_DYNAMIC_RANGE:
    db_max = filtered.max()
    db_min = filtered.min()

    # data‑driven color stretch
    vmin = db_max - 20
    vmax = db_max - 1

    plt.imshow(
        filtered,
        extent=[IMAGE_CENTER_Y - TERRAIN_SIZE / 2, IMAGE_CENTER_Y + TERRAIN_SIZE / 2, IMAGE_CENTER_X - TERRAIN_SIZE / 2, IMAGE_CENTER_X + TERRAIN_SIZE / 2],
        origin='lower',
        cmap='viridis',
        vmin=vmin, vmax=vmax
    )
else:
    plt.imshow(
        filtered,
        extent=[IMAGE_CENTER_Y - TERRAIN_SIZE / 2, IMAGE_CENTER_Y + TERRAIN_SIZE / 2, IMAGE_CENTER_X - TERRAIN_SIZE / 2, IMAGE_CENTER_X + TERRAIN_SIZE / 2],
        origin='lower',
        cmap='viridis', 
        vmin=130, vmax=150
    )
plt.colorbar(label='Amplitude (dB)')
plt.title('SAR Backprojection, Numpy (db)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')

if SHOW_DETECTED_PEAKS:
    # Plot detected blobs as scatter points
    plt.scatter(blob_y_world, blob_x_world,  color='red', marker='x', label='Blobs')

    # Display the scale (sigma) next to each detected blob
    for (xw, yw, (_, _, sigma)) in zip(blob_x_world, blob_y_world, blobs):
        plt.text(yw,xw, f"{sigma:.1f}", color='white', fontsize=8, ha='left', va='bottom')

    plt.legend()
plt.show()

# =============================================================================
# COMPARISON WINDOW: Show all alignment variations side by side
# =============================================================================

def process_pickle_file(pickle_path, title_suffix=""):
    """Process a single pickle file and return the backprojection image"""
    try:
        scan_records = open_pickle_file(pickle_path)
        if scan_records is None:
            return None, f"Failed to load: {title_suffix}"
        
        print(f"  Data shapes: platform_pos={scan_records['platform_pos'].shape}, scan_data={scan_records['scan_data'].shape}")
        
        # Use same processing as main script
        radar_pos_arr = np.asarray(scan_records['platform_pos'])
        amplitudes_arr = np.asarray(scan_records['scan_data'])
        range_bins_arr = np.asarray(scan_records['range_bins'])
        
        # Same grid as main script
        x = np.arange(IMAGE_CENTER_X - TERRAIN_SIZE / 2, IMAGE_CENTER_X + TERRAIN_SIZE / 2, TERRAIN_RESOLUTION)
        y = np.arange(IMAGE_CENTER_Y - TERRAIN_SIZE / 2, IMAGE_CENTER_Y + TERRAIN_SIZE / 2, TERRAIN_RESOLUTION)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        backproj_img = np.zeros_like(X, dtype=np.complex128)
        pixel_pos = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        num_pixels = pixel_pos.shape[0]
        num_scans = len(scan_records['scan_data'])
        
        # Use same threading approach but with fewer scans for speed
        def process_chunk_comparison(chunk_indices):
            chunk_pixel_pos = pixel_pos[chunk_indices]
            chunk_sum = np.zeros(len(chunk_indices), dtype=np.complex128)
            
            # Use every 10th scan for comparison (still gets good image)
            for scan_idx in range(0, num_scans, 10):
                radar_pos = radar_pos_arr[scan_idx]
                range_bin = range_bins_arr
                dists = np.linalg.norm(chunk_pixel_pos - radar_pos, axis=1)
                
                if HILBERT:
                    amp = retrieve_complex_data(amplitudes_arr[scan_idx], range_bin)
                    amp = amp.T[0]
                    amp_real = np.interp(dists, range_bin, amp.real)
                    amp_imag = np.interp(dists, range_bin, amp.imag)
                    interp_amp = amp_real + 1j * amp_imag
                    interp_amp *= np.exp(4j * np.pi * dists * 4.3e9 / 2.99792458e8)
                else:
                    interp_amp = np.interp(dists, range_bin, amplitudes_arr[scan_idx])
                
                chunk_sum += interp_amp
            
            return chunk_indices, chunk_sum
        
        # Process in chunks (smaller chunks for speed)
        chunk_size = num_pixels // 8  # 8 chunks instead of 18
        chunks = [np.arange(i, min(i + chunk_size, num_pixels)) for i in range(0, num_pixels, chunk_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_chunk_comparison, chunks))
        
        # Combine results
        pixel_results = np.zeros(num_pixels, dtype=np.complex128)
        for chunk_indices, chunk_sum in results:
            pixel_results[chunk_indices] = chunk_sum
        
        backproj_img = pixel_results.reshape(X.shape)
        backproj_magnitude = np.abs(backproj_img)
        backproj_db = 20 * np.log10(backproj_magnitude + 1e-12)
        
        print(f"  DB range: {np.min(backproj_db):.1f} to {np.max(backproj_db):.1f}")
        
        if AUTO_DYNAMIC_RANGE:
            filtered = gaussian_filter(backproj_db, sigma=1)
        else:
            filtered = backproj_db
            
        return filtered, title_suffix
        
    except Exception as e:
        print(f"Error processing {pickle_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {title_suffix}"

print("\nGenerating comparison plots...")

# Define the files to compare
base_dir = r"c:\Users\soham\Coding Projects\Team_Four_Repo-Simple\Aug1Flight1"
comparison_files = [
    (os.path.join(base_dir, "aligned_data.pkl"), "Main aligned_data.pkl"),
    (os.path.join(base_dir, "BATCH", "-60ms.pkl"), "-60ms"),
    (os.path.join(base_dir, "BATCH", "-30ms.pkl"), "-30ms"), 
    (os.path.join(base_dir, "BATCH", "+0ms.pkl"), "+0ms"),
    (os.path.join(base_dir, "BATCH", "+30ms.pkl"), "+30ms"),
    (os.path.join(base_dir, "BATCH", "+60ms.pkl"), "+60ms")
]

# Create comparison figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Alignment Timing Comparison - Aug1Flight1', fontsize=16)

# First pass: process all images to find global min/max for consistent scaling
print("First pass: finding dynamic range...")
all_images = []
all_titles = []
for file_path, title in comparison_files:
    print(f"Processing {title}...")
    filtered_img, actual_title = process_pickle_file(file_path, title)
    all_images.append(filtered_img)
    all_titles.append(actual_title)

# Find global range for consistent scaling
valid_images = [img for img in all_images if img is not None]
if valid_images:
    global_min = min(np.min(img) for img in valid_images)
    global_max = max(np.max(img) for img in valid_images)
    # Use percentiles for better dynamic range
    all_values = np.concatenate([img.flatten() for img in valid_images])
    vmin = np.percentile(all_values, 5)  # 5th percentile
    vmax = np.percentile(all_values, 95)  # 95th percentile
    print(f"Using dynamic range: {vmin:.1f} to {vmax:.1f} dB")
else:
    vmin, vmax = 0, 1

# Second pass: plot with consistent scaling
for idx, (filtered_img, actual_title) in enumerate(zip(all_images, all_titles)):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    if filtered_img is not None:
        im = ax.imshow(
            filtered_img,
            extent=[IMAGE_CENTER_Y - TERRAIN_SIZE / 2, IMAGE_CENTER_Y + TERRAIN_SIZE / 2, 
                   IMAGE_CENTER_X - TERRAIN_SIZE / 2, IMAGE_CENTER_X + TERRAIN_SIZE / 2],
            origin='lower',
            cmap='viridis',
            vmin=vmin, vmax=vmax
        )
        ax.set_title(actual_title, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # Add colorbar to each subplot
        plt.colorbar(im, ax=ax, label='dB')
    else:
        ax.text(0.5, 0.5, f'Failed to load\n{actual_title}', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title(f'Error: {actual_title}', fontsize=12)

plt.tight_layout()
plt.show()

print("Comparison plots generated!")