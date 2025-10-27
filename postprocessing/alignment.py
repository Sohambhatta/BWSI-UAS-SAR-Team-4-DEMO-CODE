import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import json
import struct
import pickle
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import quaternion
import sys
import os
from QuaternionKalmanFilter import eskf_filter_quaternions


if len(sys.argv) > 1:
    DIRECTORY = sys.argv[1]
else:
    print("Error: Please provide a directory path as an argument.")
    DIRECTORY = "Aug1Flight1"



BATCH_RUN = True

BATCH_COUNT = 5
BATCH_TIME_STEP = 30 # milliseconds

# actual_square0_pos = np.array([2.341377, 0.139044, 2.038765])  # Square0 position 
# actual_square1_pos = np.array([-0.874452, 0.189028, -1.703920])  # Square1 position July 29

#actual_square0_pos = np.array([2.30036, 0.138837, 2.052387])  # Square0 position 
#actual_square1_pos = np.array([-0.862372, 0.192635, -1.757383])  # Square1 position July 30

# actual_square0_pos = np.array([2.267839, 0.151220, 2.070158])  # Square0 position 
# actual_square1_pos = np.array([-0.903028, 0.184394, -1.817815])  # Square1 position July 31


actual_square0_pos = np.array([-0.950737, 0.1863414, -1.986565])  # Square0 position 
actual_square1_pos = np.array([2.37161152, 0.154459, 2.112262])  # Square1 position Aug 1


DX_SEARCH_MIN = 2.95  # Minimum dx search range in meters
DX_SEARCH_MAX = 3.01  # Maximum dx search range in meters
DX_SEARCH_COARSE_STEP = 0.01  # Coarse step size for dx search in meters
DX_SEARCH_FINE_STEP = 0.01  # Fine step size for dx search in meters

DT_SEARCH_MIN = -10000  # Minimum dt search range in milliseconds
DT_SEARCH_MAX = 0  # Maximum dt search range in milliseconds
DT_SEARCH_COARSE_STEP = 20  # Coarse step size for dt search in milliseconds
DT_SEARCH_FINE_STEP = 1  # Fine step size for dt search in milliseconds

APPLY_QUATERNION = True  # Whether to apply quaternion adjustment to positions
DT_OFFSET = +40
DX_OFFSET = 0.0  # Default dx offset in meters, can be adjusted later

def adjust_pos_with_quaternion(i, j, k, r, x, y, z):
    """
    Adjusts the position based on quaternion rotation.
    Args:
        i, j, k, r: Quaternion components
        x, y, z: Position coordinates
    Returns:
        Adjusted position as a numpy array.
    """
    
    q = np.quaternion(r, i, j, k)  # Create quaternion
    q = q / np.abs(q)  # Normalize the quaternion
    q_conjugate = q.conjugate()  # Conjugate of the quaternion
    q_conjugate = q_conjugate / np.abs(q_conjugate)  # Normalize the conjugate quaternion
    drone_boresight = np.quaternion(0, 0.125, -0.15, 0)  # Boresight vector
    
    if not APPLY_QUATERNION:
        drone_boresight = np.quaternion(0, 0, 0, 0)
    world_vector = q * drone_boresight * q.conjugate()  # Rotate boresight vector
    #print(world_vector)
    adjusted_pos = np.quaternion(0, x, y, z) + world_vector  # Adjust position
    adjusted_pos = np.array([adjusted_pos.x, adjusted_pos.y, adjusted_pos.z])  # Convert to numpy array
    #adjusted_pos = [x, y, z]
    return adjusted_pos

def plot_distances_to_squares(pos_csv_dir):
    """
    Reads drone position CSV file and computes distances to two squares.
    Returns:
        dist_arr: np.ndarray of shape (num_frames, 2) with distances to square 0 and square 1
        time_arr: np.ndarray of shape (num_frames,) with timestamps in seconds
    """
    pos_csv_path = os.path.join(pos_csv_dir, "pos.csv")
    frames = []
    times = []
    xs = []
    ys = []
    zs = []
    quats = []
    with open(pos_csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 0 and row[0].strip().lower() == 'frame':
                break
        for row in reader:
            if len(row) < 9 or row[6] == '' or row[7] == '' or row[8] == '':
                continue
            frame, time, i, j, k, r, x, y, z = row[:9]
            frames.append(int(frame))
            times.append(float(time))
            quats.append([float(r), float(i), float(j), float(k)])
            xs.append(float(x))
            ys.append(float(y))
            zs.append(float(z))
    # Filter quaternion series
    quats = np.array(quats, dtype=np.float64)
    filtered_quats, outliers, innovations = eskf_filter_quaternions(quats, dt = 1/360, process_noise = 1e-3, measurement_noise=5e-3)
    i_filt = filtered_quats[:, 1]
    j_filt = filtered_quats[:, 2]
    k_filt = filtered_quats[:, 3]
    r_filt = filtered_quats[:, 0]
    # Apply rotation
    positions = np.array([
        adjust_pos_with_quaternion(i_filt[idx], j_filt[idx], k_filt[idx], r_filt[idx], xs[idx], ys[idx], zs[idx])
        for idx in range(len(xs))
    ])
    dist0 = np.linalg.norm(positions - actual_square0_pos, axis=1)
    dist1 = np.linalg.norm(positions - actual_square1_pos, axis=1)
    dist_arr = np.vstack([dist0, dist1]).T
    time_arr = np.array(times)
    sort_idx = np.argsort(time_arr)
    dist_arr = dist_arr[sort_idx]
    time_arr = time_arr[sort_idx]
    return dist_arr, time_arr

def parse_rti_file(file_path):
    """
    Parses a binary RTI file using accompanying metadata.json.
    Returns:
        time_axis: np.ndarray of timestamps in milliseconds
        range_axis: np.ndarray of computed range bins in meters
        intensity: np.ndarray of shape (num_ranges, num_times)
    """
    import numpy as np, os, json
    metadata_path = os.path.join(file_path, "metadata.json")
    binary_path = os.path.join(file_path, "returns.bin")
    picolightseconds = 0.000299792458  # meters per picosecond
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    scan_start = metadata["scan_start"]
    scan_res = metadata["scan_res"]
    arr = np.fromfile(binary_path, dtype=np.int32)
    time_stamps = []
    amplitude_rows = []
    i = 0
    while i < len(arr):
        num_elements = arr[i]
        millis_time = arr[i+1]
        time_stamps.append(millis_time)
        amplitudes = arr[i+2:i+2+num_elements]
        amplitude_rows.append(amplitudes)
        i += 2 + num_elements
    time_axis = np.array(time_stamps, dtype=np.uint32)
    max_len = max(len(row) for row in amplitude_rows)
    intensity = np.zeros((max_len, len(amplitude_rows)), dtype=np.int32)
    for t_idx, row in enumerate(amplitude_rows):
        intensity[:len(row), t_idx] = row
    range_axis = scan_start * picolightseconds / 2 + \
                 np.arange(max_len) * (scan_res * 1.907 * picolightseconds / 2)
    return time_axis, range_axis, intensity

def find_best_dt_dx_scipy(radar_time_ms, ranges, intensity_db_clipped, dist_arr, pos_time_ms):
    """
    Finds the best time offset (dt) and range offset (dx) to align drone positions with RTI data.
    Uses a coarse and fine search strategy to optimize the alignment score.
    Args:
        radar_time_ms: np.ndarray of radar timestamps in milliseconds
        ranges: np.ndarray of range bins in meters
        intensity_db_clipped: np.ndarray of intensity values in dB, shape (num_ranges, num_frames)
        dist_arr: np.ndarray of shape (num_frames, 2) with distances to square 0 and square 1
        pos_time_ms: np.ndarray of drone position timestamps in milliseconds
    Returns: 
        best_dt: int, best time offset in milliseconds
        best_dx: float, best range offset in meters
    """

    # gaussian filter to smooth the intensity data
    intensity_db_smooth = gaussian_filter1d(intensity_db_clipped, sigma=2, axis=1)
    
    best_dt = 0
    best_dx = 0
    best_score = -np.inf

    # Coarse search
    coarse_dt_range = np.arange(DT_SEARCH_MIN, DT_SEARCH_MAX, DT_SEARCH_COARSE_STEP)
    coarse_dx_range = np.arange(DX_SEARCH_MIN, DX_SEARCH_MAX, DX_SEARCH_COARSE_STEP)
    for dt in coarse_dt_range:
        #steps through coarse dt range to align time
        shifted_time = pos_time_ms + dt
        for dx in coarse_dx_range:
            # optimize dx
            score = 0
            f_time = interp1d(radar_time_ms, np.arange(len(radar_time_ms)), bounds_error=False, fill_value="extrapolate")
            f_range = interp1d(ranges, np.arange(len(ranges)), bounds_error=False, fill_value="extrapolate")
            t_idx = f_time(shifted_time).astype(int)
            # Both curves offset by dx
            r_idx0 = f_range(dist_arr[:, 0] + dx).astype(int)
            r_idx1 = f_range(dist_arr[:, 1] + dx).astype(int)
            t_idx = np.clip(t_idx, 0, intensity_db_smooth.shape[1] - 1)
            r_idx0 = np.clip(r_idx0, 0, intensity_db_smooth.shape[0] - 1)
            r_idx1 = np.clip(r_idx1, 0, intensity_db_smooth.shape[0] - 1)
            score += np.sum(intensity_db_smooth[r_idx0, t_idx])
            score += np.sum(intensity_db_smooth[r_idx1, t_idx])
            if score > best_score:
                best_score = score
                best_dt = dt
                best_dx = dx

    # Fine search:
    fine_dt_range = np.arange(best_dt - DT_SEARCH_COARSE_STEP, best_dt + DT_SEARCH_COARSE_STEP + 1, DT_SEARCH_FINE_STEP)
    fine_dx_range = np.arange(best_dx - DX_SEARCH_COARSE_STEP, best_dx + DX_SEARCH_COARSE_STEP * 1.1, DX_SEARCH_FINE_STEP)
    for dt in fine_dt_range:
        # steps through fine dt range to align time
        shifted_time = pos_time_ms + dt
        for dx in fine_dx_range:
            # optimize dx
            score = 0
            f_time = interp1d(radar_time_ms, np.arange(len(radar_time_ms)), bounds_error=False, fill_value="extrapolate")
            f_range = interp1d(ranges, np.arange(len(ranges)), bounds_error=False, fill_value="extrapolate")
            t_idx = f_time(shifted_time).astype(int)
            r_idx0 = f_range(dist_arr[:, 0] + dx).astype(int)
            r_idx1 = f_range(dist_arr[:, 1] + dx).astype(int)
            t_idx = np.clip(t_idx, 0, intensity_db_smooth.shape[1] - 1)
            r_idx0 = np.clip(r_idx0, 0, intensity_db_smooth.shape[0] - 1)
            r_idx1 = np.clip(r_idx1, 0, intensity_db_smooth.shape[0] - 1)
            score += np.sum(intensity_db_smooth[r_idx0, t_idx])
            score += np.sum(intensity_db_smooth[r_idx1, t_idx])
            if score > best_score:
                best_score = score
                best_dt = dt
                best_dx = dx
    return best_dt, best_dx


def save_aligned_pickle(rti_dir, pos_csv_dir, dt, dx):
    # Load RTI data
    radar_time_ms, ranges, intensity = parse_rti_file(rti_dir)
    # Load drone positions
    dist_arr, time_arr = plot_distances_to_squares(pos_csv_dir)
    xs, ys, zs = [], [], []
    pos_csv_path = os.path.join(pos_csv_dir, "pos.csv")
    quats = []
    with open(pos_csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 0 and row[0].strip().lower() == 'frame':
                break
        for row in reader:
            if len(row) < 9 or row[6] == '' or row[7] == '' or row[8] == '':
                continue
            frame, time, i, j, k, r, x, y, z = row[:9]
            quats.append([float(r), float(i), float(j), float(k)])
            xs.append(float(x))
            ys.append(float(y))
            zs.append(float(z))
    # Filter quaternion series
    quats = np.array(quats, dtype=np.float64)
    filtered_quats, outliers, innovations = eskf_filter_quaternions(quats, dt = 1/360, process_noise = 1e-3, measurement_noise=5e-3)
    i_filt = filtered_quats[:, 1]
    j_filt = filtered_quats[:, 2]
    k_filt = filtered_quats[:, 3]
    r_filt = filtered_quats[:, 0]
    
    positions = np.array([
        adjust_pos_with_quaternion(i_filt[idx], j_filt[idx], k_filt[idx], r_filt[idx], xs[idx], ys[idx], zs[idx])
        for idx in range(len(xs))
    ])

    #positions = np.column_stack([xs, ys, zs])
    pos_time_ms = time_arr * 1000

    # Adjust range bins with dx
    range_bins = ranges - dx  # 1D np.ndarray

    # Interpolate platform positions for each scan time (with dt applied)
    scan_times_ms = radar_time_ms - radar_time_ms[0] - dt
    
    print(scan_times_ms)
    interp_x = interp1d(pos_time_ms, positions[:, 0], bounds_error=False, fill_value="extrapolate")
    interp_y = interp1d(pos_time_ms, positions[:, 1], bounds_error=False, fill_value="extrapolate")
    interp_z = interp1d(pos_time_ms, positions[:, 2], bounds_error=False, fill_value="extrapolate")
    platform_pos = np.column_stack([
        interp_x(scan_times_ms),
        interp_z(scan_times_ms),
        interp_y(scan_times_ms)
    ])  # shape (num_scans, 3), np.ndarray

    # scan_data: shape (num_scans, num_amplitudes), np.ndarray
    scan_data = intensity.T  # shape (num_scans, num_amplitudes)

    # Save to pickle
    out_dict = {
        'platform_pos': platform_pos,
        'scan_data': scan_data,
        'range_bins': range_bins
    }
    out_path = os.path.join(rti_dir, "aligned_data.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out_dict, f)
    print(f"Aligned pickle saved to {out_path}")

def plot_rti_with_distances(rti_dir, pos_csv_dir):
    # Load RTI data
    radar_time_ms, ranges, intensity = parse_rti_file(rti_dir)
    intensity_db = 20 * np.log10(np.abs(intensity) + 1e-12) # Convert to dB scale
    max_db = np.max(intensity_db) # Find the maximum dB value in the data
    intensity_db_clipped = np.copy(intensity_db) # Set all values below (max_db - 100) to zero (ignore them)
    intensity_db_clipped[intensity_db_clipped < (max_db - 100)] = 0 

    # Load drone positions and distances
    dist_arr, time_arr = plot_distances_to_squares(pos_csv_dir)
    # Convert time_arr from seconds to milliseconds for overlay
    pos_time_ms = time_arr * 1000
    #print(pos_time_ms)
    radar_time_ms = radar_time_ms - radar_time_ms[0]  # Normalize radar time to start from 0
    #print(radar_time_ms)

    # Find best dt and dx using scipy tools
    dt, dx = find_best_dt_dx_scipy(radar_time_ms, ranges, intensity_db_clipped, dist_arr, pos_time_ms)
    dt += DT_OFFSET  # Apply offset to dt
    dx += DX_OFFSET  # Apply offset to dx

    print(f"Best dt (ms) for alignment: {dt}")
    print(f"Best dx (m) for alignment: {dx}")

    # Save aligned pickle file
    save_aligned_pickle(rti_dir, pos_csv_dir, dt, dx)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        intensity_db_clipped.T,
        aspect='auto',
        extent=[ranges[0], ranges[-1], radar_time_ms[-1], radar_time_ms[0]],
        cmap='viridis',
        interpolation='nearest',
        vmin=35
    )
    plt.colorbar(label='dB')
    plt.xlabel('Range (m)')
    plt.ylabel('Time (ms)')
    plt.title('RTI Plot with Drone-Square Distances (Aligned)')

    # Both curves offset by dx and dt
    plt.plot(dist_arr[:, 0] + dx, pos_time_ms + dt, label="Square 0 (corner, aligned)", color="blue", linewidth=2)
    plt.plot(dist_arr[:, 1] + dx, pos_time_ms + dt, label="Square 1 (corner, aligned)", color="red", linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print("RTI alignment and manual tuning tool")
    rti_dir = DIRECTORY
    pos_csv_dir = DIRECTORY

    # Load RTI data
    radar_time_ms, ranges, intensity = parse_rti_file(rti_dir)
    intensity_db = 20 * np.log10(np.abs(intensity) + 1e-12)
    max_db = np.max(intensity_db)
    intensity_db_clipped = np.copy(intensity_db)
    intensity_db_clipped[intensity_db_clipped < (max_db - 100)] = 0
    dist_arr, time_arr = plot_distances_to_squares(pos_csv_dir)
    pos_time_ms = time_arr * 1000
    radar_time_ms = radar_time_ms - radar_time_ms[0]

    # Always run autoalign first
    best_dt, best_dx = find_best_dt_dx_scipy(radar_time_ms, ranges, intensity_db_clipped, dist_arr, pos_time_ms)
    print(f"Automatic alignment complete.")
    print(f"Best dt (ms): {best_dt}")
    print(f"Best dx (m): {best_dx}")

    if BATCH_RUN:
        batch_dir = os.path.join(rti_dir, "BATCH")
        os.makedirs(batch_dir, exist_ok=True)
        # Empty the batch directory before saving new pkls
        for fname in os.listdir(batch_dir):
            fpath = os.path.join(batch_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
        center_idx = BATCH_COUNT // 2
        for i in range(BATCH_COUNT):
            offset = (i - center_idx) * BATCH_TIME_STEP
            dt_batch = best_dt + offset
            dx_batch = best_dx
            out_name = f"{offset:+d}ms.pkl"
            out_path = os.path.join(batch_dir, out_name)
            print(f"Saving batch pickle: {out_name} (dt={dt_batch}, dx={dx_batch})")
            save_aligned_pickle(rti_dir, pos_csv_dir, dt_batch, dx_batch)
            orig_pkl = os.path.join(rti_dir, "aligned_data.pkl")
            if os.path.exists(orig_pkl):
                os.replace(orig_pkl, out_path)
        # Only graph the original autoalign
        dt = best_dt
        dx = best_dx
        print(f"Plotting original autoalign dt={dt}, dx={dx}")
        save_aligned_pickle(rti_dir, pos_csv_dir, dt, dx)
        plt.figure(figsize=(10, 6))
        plt.imshow(
            intensity_db_clipped.T,
            aspect='auto',
            extent=[ranges[0], ranges[-1], radar_time_ms[-1], radar_time_ms[0]],
            cmap='viridis',
            interpolation='nearest',
            vmin=35
        )
        plt.colorbar(label='dB')
        plt.xlabel('Range (m)')
        plt.ylabel('Time (ms)')
        plt.title('RTI Plot with Drone-Square Distances (Aligned)')
        plt.plot(dist_arr[:, 0] + dx, pos_time_ms + dt, label="Square 0 (corner, aligned)", color="blue", linewidth=2)
        plt.plot(dist_arr[:, 1] + dx, pos_time_ms + dt, label="Square 1 (corner, aligned)", color="red", linewidth=2)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

main()