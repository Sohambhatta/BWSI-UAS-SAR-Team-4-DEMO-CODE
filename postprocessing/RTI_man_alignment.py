
import sys
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QMainWindow, QWidget, QHBoxLayout, QSlider
from PySide6.QtCore import Slot, Qt
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
import os
import json
import struct
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import csv
import pickle
from scipy.interpolate import interp1d

loc = [-0.950737, 0.1863414, -1.986565]

DIR = 'DATAS/July28Slide4/'  # Change this to your actual CSV file path

def parse_csv(pos_csv_dir):
    """
    Reads drone positions from CSV and computes distance to a single target position.
    Args:
        pos_csv_dir: directory containing 'pos.csv'
        target_position: tuple of (x, y, z) for the target
    Returns:
        times: np.ndarray of timestamps in seconds
        distances: np.ndarray of shape (num_frames,) distances to the target
        positions: np.ndarray of shape (num_frames, 3) positions of the drone
    """
    DIRECTORY = os.path.join(pos_csv_dir, 'pos.csv')
    x_targ, y_targ, z_targ = [], [], []
    x_plat, y_plat, z_plat = [], [], []


    with open(DIRECTORY, "r") as f:
        reader = csv.reader(f)
        # Skip header lines until we reach the data
        for _ in range(8):
            next(reader)
        for row in reader:
            if len(row) < 7 or row[1] == '':
                continue
            # try:
            if True:
                if (row[6] != '' and row[7] != '' and row[8] != ''):
                    times.append(float(row[1]))

                    # x_targ.append(float(row[2]) if row[2] else np.nan)
                    # y_targ.append(float(row[3]) if row[3] else np.nan)
                    # z_targ.append(float(row[4]) if row[4] else np.nan)
                    x_targ.append(loc[0])
                    y_targ.append(loc[1])
                    z_targ.append(loc[2])
                    x_plat.append(float(row[6]))
                    y_plat.append(float(row[7]))
                    z_plat.append(float(row[8]))
            # except Exception:
            #     print(Exception)
            #     continue
    # Convert to numpy arrays
    times = np.array(times)
    x_targ = np.array(x_targ)
    y_targ = np.array(y_targ)
    z_targ = np.array(z_targ)
    x_plat = np.array(x_plat)
    y_plat = np.array(y_plat)
    z_plat = np.array(z_plat)

    # Compute distances
    targ_pos = np.stack([x_targ, y_targ, z_targ], axis=1)
    plat_pos = np.stack([x_plat, y_plat, z_plat], axis=1)
    distances = np.sqrt(
        (x_targ - x_plat) ** 2 +
        (y_targ - y_plat) ** 2 +
        (z_targ - z_plat) ** 2
    )
    return times, distances, plat_pos
        


#part one:  reading the data from the file
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
def apply_median_filter(intensity_matrix, size=(3, 3)):
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
def threshold(intensity_matrix, threshold_db=-80):
    """
    Zeroes out values below the threshold.
    """
    filtered = np.copy(intensity_matrix)
    filtered[filtered < threshold_db] = 0
    return filtered


# Part 1.5: moving average filter
def moving_average(intensity_matrix, window=3):
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



def moving_average_along_time(intensity_matrix, window=3):
    # Apply moving average along each row (range bin)
    result = np.zeros_like(intensity_matrix)
    for i in range(intensity_matrix.shape[0]):
        result[i, :] = np.convolve(intensity_matrix[i, :], np.ones(window)/window, mode='same')
    return result





# Allow user to specify just the subdirectory name (e.g., 'July22Slide1')
data_dir = DIR

time_ms, ranges, intensity = parse_rti_file(data_dir)
intensity_db = 20 * np.log10(np.abs(intensity) + 1e-12)

# Find the maximum dB value in the data
max_db = np.max(intensity_db)

# Set all values below (max_db - 100) to zero (ignore them)
intensity_db_clipped = np.copy(intensity_db)
intensity_db_clipped[intensity_db_clipped < (max_db - 100)] = 0
time_ms2, intensity2 = remove_most_common_frame(intensity_db_clipped, time_ms)


# Set up the GUI
class RTIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTI Postprocessing")

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Button row
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        #Slider setup
        slider1_layout = QVBoxLayout()
        content_layout.addLayout(slider1_layout)

        slider1 = QSlider(Qt.Vertical)
        slider1_layout.addWidget(slider1)
        slider1.setRange(-15000, 40000)
        slider1.setValue(0)

        
        slider2_layout = QHBoxLayout()
        main_layout.addLayout(slider2_layout)
        
        slider2 = QSlider(Qt.Horizontal)
        slider2_layout.addWidget(slider2)
        slider2.setRange(1000, 8000)
        slider2.setValue(5500)

        rtn_button = QPushButton("Return Aligned Data")
        rtn_button.clicked.connect(self.return_aligned_data)
        slider2_layout.addWidget(rtn_button)

        # Matplotlib Figure and Canvas
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        content_layout.addWidget(self.canvas)

        # Add buttons
        buttons = [
            ("Original Plot", self.original_plot),
            ("Remove Most Common Frame", self.remove_most_common_frame_RTI),
            ("Subtract Initial Frames", self.subtract_initial_frames_RTI),
            ("Median Filter", self.median_filter_RTI),
            ("Threshold", self.threshold_RTI),
            ("Mask Static Ranges", self.mask_static_ranges_RTI),
            ("Moving Average", self.moving_average_RTI),
        ]
        for label, func in buttons:
            btn = QPushButton(label)
            btn.clicked.connect(func)
            button_layout.addWidget(btn)

        self.theoretical_times, self.theoretical_distances, self.platform_pos = parse_csv(data_dir)
        print(self.platform_pos.shape)
        self.time_adjust = slider1.value()
        self.range_adjust = slider2.value()

        # Interpolate theoretical distances using the full range of theoretical_times
        # Create interp_times that covers the theoretical_times range at the RTI time spacing
        rti_time_spacing = np.median(np.diff(time_ms2))
        interp_start = (self.theoretical_times[0] - self.theoretical_times[0]) * 1000  # 0
        interp_end = (self.theoretical_times[-1] - self.theoretical_times[0]) * 1000
        num_interp = int(np.round((interp_end - interp_start) / rti_time_spacing)) + 1
        self.interp_times = np.linspace(interp_start, interp_end, num_interp)

        # Interpolate theoretical distances to this spacing
        self.interp_theoretical_distances = np.interp(
            self.interp_times,
            (self.theoretical_times - self.theoretical_times[0]) * 1000,  # convert to ms, start at 0
            self.theoretical_distances
        )
        self.rti_time_axis = time_ms2  # Keep for plotting, but interpolation is by spacing

        slider1.valueChanged.connect(self.update_theoretical_track)
        slider2.valueChanged.connect(self.update_theoretical_track)
        self.slider1 = slider1
        self.slider2 = slider2

        # Initial plot
        self.original_plot()  # Show the original plot on startup
    
    # Different Plotting options 
    @Slot()
    def original_plot(self):
        # 1. Plot original
        self.plot_rti_gui(time_ms, ranges, intensity_db_clipped)

    @Slot()
    def remove_most_common_frame_RTI(self):
        # 2. Remove most common frame
        time_ms2, intensity2 = remove_most_common_frame(intensity_db_clipped, time_ms)
        self.plot_rti_gui(time_ms2, ranges, intensity2)

    @Slot()
    def subtract_initial_frames_RTI(self):
        # 3. Subtract initial frames
        intensity_bgsub = subtract_initial_frames(intensity_db_clipped, num_frames=3)
        self.plot_rti_gui(time_ms2, ranges, intensity_bgsub)

    @Slot()
    def median_filter_RTI(self):
        # 4. Median filter
        intensity_median = apply_median_filter(intensity_db_clipped, size=(3, 3))
        self.plot_rti_gui(time_ms2, ranges, intensity_median)

    @Slot()
    def threshold_RTI(self):
        # 5. Thresholding
        intensity_thresh = threshold(intensity_db_clipped, threshold_db=-80)
        self.plot_rti_gui(time_ms2, ranges, intensity_thresh)

    @Slot()
    def mask_static_ranges_RTI(self):
        # 6. Mask static ranges
        intensity_masked = mask_static_ranges(intensity_db_clipped)
        self.plot_rti_gui(time_ms2, ranges, intensity_masked)

    @Slot()
    def moving_average_RTI(self):
        # 7. Moving average (along time axis for each range bin)
        intensity_ma = moving_average_along_time(intensity_db_clipped, window=3)
        self.plot_rti_gui(time_ms2, ranges, intensity_ma)
    
    def plot_rti_gui(self, time_ms, ranges, intensity_matrix, title="RTI Plot (Range-Time Intensity)"):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        cax = self.ax.imshow(
            intensity_matrix.T,
            aspect='auto',
            extent=[ranges[0], ranges[-1], time_ms[-1], time_ms[0]],
            cmap='viridis',
            interpolation='nearest'
        )
        self._colorbar = self.fig.colorbar(cax, ax=self.ax, label='dB')
        self.ax.set_xlabel('Range (m)')
        self.ax.set_ylabel('Time (ms)')
        self.ax.set_title(title)
        self.ax.set_xlim(ranges[0], ranges[-1])
        self.ax.set_ylim(time_ms[-1], time_ms[0])
        self.canvas.draw()


    def overlay_distances(self, time_axis, distances, labels):
        for i, label in enumerate(labels):
            # Rescale distances to overlay on plot (normalize if needed)
            scaled_distances = distances[:, i]
            # Match time axis
            self.ax.plot(scaled_distances, time_axis, label=label)
        self.ax.legend()

    def update_theoretical_track(self):
        self.time_adjust = -self.slider1.value()
        self.range_adjust = self.slider2.value()
        self.update_theoretical_overlay()

    def update_theoretical_overlay(self):
        # Remove previous theoretical track if it exists
        if hasattr(self, '_theoretical_line') and self._theoretical_line is not None:
            self._theoretical_line.remove()
            self._theoretical_line = None

        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        # Interpolate theoretical distances to RTI time axis (ms)
        self.rti_time_axis = self.rti_time_axis.astype(int)  # Ensure float for addition
        adjusted_times = self.interp_times + self.time_adjust + self.rti_time_axis[0]  # Adjust by RTI start time
        adjusted_ranges = self.interp_theoretical_distances + self.range_adjust / 1000

        mask = (
            (adjusted_ranges >= min(x_min, x_max)) & (adjusted_ranges <= max(x_min, x_max)) &
            (adjusted_times >= min(y_min, y_max)) & (adjusted_times <= max(y_min, y_max))
        )
        [line] = self.ax.plot(adjusted_ranges[mask], adjusted_times[mask], color='red', label='Theoretical Track')
        self._theoretical_line = line
        self.ax.legend()
        self.canvas.draw()

    def return_aligned_data(self):
        """
        Returns the aligned RTI data as a dictionary.
        """
        y_min, y_max = self.ax.get_ylim()
        # Use interpolated and adjusted times/ranges
        #aligned_times = self.rti_time_axis + self.time_adjust
        aligned_ranges = ranges + self.range_adjust / 1000
        #print("x")
        #print(aligned_ranges.shape)
        pos_time_ms = self.theoretical_times * 1000.0

        aligned_times = time_ms.astype(int) - self.time_adjust - int(time_ms[0])  # Adjust by RTI start time
        #print(aligned_times.shape)
        interp_x = interp1d(pos_time_ms, self.platform_pos[:,0], bounds_error=False, fill_value="extrapolate")
        interp_y = interp1d(pos_time_ms, self.platform_pos[:,1], bounds_error=False, fill_value="extrapolate")
        interp_z = interp1d(pos_time_ms, self.platform_pos[:,2], bounds_error=False, fill_value="extrapolate")

        platform_p = np.column_stack([
        interp_x(aligned_times),
        interp_z(aligned_times),
        interp_y(aligned_times)
        ])



        global intensity
        scan_data = intensity.T

        #print(platform_p.shape)
        #print(scan_data.shape)
        #print(aligned_times.shape)

        print(self.time_adjust, self.range_adjust)

        out_dict = {
            "platform_pos": platform_p,
            "scan_data": scan_data,
            "range_bins": aligned_ranges
        }
        out_path = "aligned_data.pkl"
        
        with open(out_path, "wb") as f:
            pickle.dump(out_dict, f)
        print(f"Aligned pickle saved to {out_path}")

        #return aligned_times, aligned_ranges




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RTIMainWindow()
    window.show()
    sys.exit(app.exec())
