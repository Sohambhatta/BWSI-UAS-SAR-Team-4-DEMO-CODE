import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
import os
import json
import struct

#part one:  reading the data from the file
def parse_rti_file(file_path):
    """
    Parses a binary RTI file using accompanying metadata.json.

    Returns:
        time_axis: np.ndarray of timestamps in milliseconds
        range_axis: np.ndarray of computed range bins in meters
        intensity: np.ndarray of shape (num_ranges, num_times)
    """
    metadata_path = os.path.join(file_path, "metadata.json")
    binary_path = os.path.join(file_path, "returns.bin")

    # Constants
    picolightseconds = 0.000299792458  # meters per picosecond

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    scan_start = metadata["scan_start"]
    scan_res = metadata["scan_res"]

    # Load binary file using numpy
    arr = np.fromfile(binary_path, dtype=np.int32)
    print(f"Loaded {len(arr)} integers from binary file.")
    # Each scan: [num_elements, millis_time, data...]
    time_stamps = []
    amplitude_rows = []
    i = 0
    while i < len(arr):
        print(i)
        num_elements = arr[i]
        millis_time = arr[i+1]
        time_stamps.append(millis_time)
        amplitudes = arr[i+2:i+2+num_elements]
        amplitude_rows.append(amplitudes)
        i += 2 + num_elements
    print(len(amplitude_rows))
    # Convert to numpy array
    time_axis = np.array(time_stamps, dtype=np.uint32)
    max_len = max(len(row) for row in amplitude_rows)
    intensity = np.zeros((max_len, len(amplitude_rows)), dtype=np.int32)
    for t_idx, row in enumerate(amplitude_rows):
        intensity[:len(row), t_idx] = row  # pad with zeros if necessary
    # Compute range axis
    range_axis = scan_start * picolightseconds / 2 + \
                 np.arange(max_len) * (scan_res * 1.907 * picolightseconds / 2)
    return time_axis, range_axis, intensity

#Part 1.2: subtracting initial frames 
def subtract_initial_frames(intensity_matrix, num_frames=3):
    """
    Subtracts the mean of the first `num_frames` frames from all frames in the intensity matrix.
    This helps remove initial background/noise from the RTI data.
    Args:
        intensity_matrix (np.ndarray): The original intensity matrix (range x time).
        num_frames (int): Number of initial frames to use for background estimation.
    Returns:
        np.ndarray: The background-subtracted intensity matrix.
    """
    if intensity_matrix.shape[1] < num_frames:
        raise ValueError("Not enough frames to subtract background.")
    # Compute mean background from the first num_frames columns
    background = np.mean(intensity_matrix[:, :num_frames], axis=1, keepdims=True)
    # Subtract background from all frames
    return intensity_matrix - background


# Part 1.3: removing the most common frame
def median_filter_rti(intensity_matrix, size=(3, 3)):
    """
    Applies a median filter to the RTI intensity matrix.
    Args:
        intensity_matrix (np.ndarray): The RTI data.
        size (tuple): The size of the filter window (range, time).
    Returns:
        np.ndarray: The filtered intensity matrix.
    """
    return median_filter(intensity_matrix, size=size)


# Part 1.4: thresholding the RTI data
def threshold_rti(intensity_matrix, threshold_db=-80):
    """
    Zeroes out values below the threshold.
    """
    filtered = np.copy(intensity_matrix)
    filtered[filtered < threshold_db] = 0
    return filtered


# Part 1.5: moving average filter
def moving_average_rti(intensity_matrix, window=3):
    """
    Applies a moving average filter along the time axis.
    """
    return np.convolve(intensity_matrix, np.ones(window)/window, mode='same')


# Part 1.6: masking static ranges
def mask_static_ranges(intensity_matrix):
    median_per_range = np.median(intensity_matrix, axis=1, keepdims=True)
    return intensity_matrix - median_per_range


# Part 1.7: removing the most common frame
def remove_most_common_frame(intensity_matrix, time_axis):
    """
    Removes the frame (column) in the intensity matrix that repeats the most.
    Returns the new intensity matrix and time axis.
    """
    # Convert columns to tuples for hashability
    columns = [tuple(intensity_matrix[:, i]) for i in range(intensity_matrix.shape[1])]
    from collections import Counter
    col_counts = Counter(columns)
    most_common_col, _ = col_counts.most_common(1)[0]
    indices_to_remove = [i for i, col in enumerate(columns) if col == most_common_col]
    mask = np.ones(intensity_matrix.shape[1], dtype=bool)
    for idx in indices_to_remove:
        mask[idx] = False
    new_intensity = intensity_matrix[:, mask]
    new_time_axis = time_axis[mask]
    return new_time_axis, new_intensity



# part two: plotting the data!!
def plot_rti(time_ms, ranges, intensity_matrix):
    # Plot the RTI (Range-Time Intensity) data as an image
    max = np.max(intensity_matrix)
    plt.figure(figsize=(10, 6))
    plt.imshow(
        intensity_matrix.T,  # Transpose so range is X and time is Y
        aspect='auto',
        extent=[ranges[0], ranges[-1], time_ms[-1], time_ms[0]],  # flip Y for earliest at top
        cmap='viridis',
        interpolation='nearest',
        vmax=max-10,
        vmin=max-50
    )   
    plt.colorbar(label='dB')  # Label colorbar as dB instead of Amplitude
    plt.xlabel('Range (m)')

    plt.ylabel('Time (ms)')
    plt.title('RTI Plot (Range-Time Intensity)')
    plt.tight_layout()
    plt.show()



# Example usage after parsing and before plotting:
# Example usage after parsing and before plotting:



import sys
if len(sys.argv) > 1:
    path = sys.argv[1]
    DIRECTORY = path
else:
    print("Error: Please provide a directory or file path as an argument.")
    DIRECTORY = 'Aug1Flight1'  # Default path if none provided

time_ms, ranges, intensity = parse_rti_file(DIRECTORY)
intensity_db = 20 * np.log10(np.abs(intensity) + 1e-12)



# Find the maximum dB value in the data
max_db = np.max(intensity_db)

# Set all values below (max_db - 100) to zero (ignore them)
intensity_db_clipped = np.copy(intensity_db)
intensity_db_clipped[intensity_db_clipped < (max_db - 100)] = 0


# 1. Plot original
plot_rti(time_ms, ranges, intensity_db_clipped)


# 2. Remove most common frame
# time_ms2, intensity2 = remove_most_common_frame(intensity_db_clipped, time_ms)
# plot_rti(time_ms2, ranges, intensity2)


# # 3. Subtract initial frames
# intensity_bgsub = subtract_initial_frames(intensity_db_clipped, num_frames=3)
# plot_rti(time_ms2, ranges, intensity_bgsub)


# # 4. Median filter
# intensity_median = median_filter_rti(intensity_db_clipped, size=(3, 3))
# plot_rti(time_ms2, ranges, intensity_median)


# # 5. Thresholding
# intensity_thresh = threshold_rti(intensity_db_clipped, threshold_db=-80)
# plot_rti(time_ms2, ranges, intensity_thresh)


# # 6. Mask static ranges
# intensity_masked = mask_static_ranges(intensity_db_clipped)
# plot_rti(time_ms2, ranges, intensity_masked)


# # 7. Moving average (along time axis for each range bin)
# def moving_average_along_time(intensity_matrix, window=3):
#     # Apply moving average along each row (range bin)
#     result = np.zeros_like(intensity_matrix)
#     for i in range(intensity_matrix.shape[0]):
#         result[i, :] = np.convolve(intensity_matrix[i, :], np.ones(window)/window, mode='same')
#     return result

# intensity_ma = moving_average_along_time(intensity_db_clipped, window=3)
# plot_rti(time_ms2, ranges, intensity_ma)