import numpy as np
from scipy.ndimage import gaussian_filter
import torch as tc
import matplotlib.pyplot as plt
import pickle as pkl
import time
import Filter
import os
import glob

"""
This code has been vectorized for optimization purposes using pytorch. 
That is why there are some places where we deal with making sure data 
in the correct places in memory, etc. 
"""

DEVICE = tc.device(
    "mps" if tc.backends.mps.is_available() else "cpu"
)  # Making sure code and operations are done on mac's gpu known as MPS instead of its cpu
FILE = "July29Flight2BII10/aligned_data.pkl"
FC = 4.3e9
C = 2.99792458e8  # Speed of light in m/s
BATCH_DIR = "BATCH"
SHOW_BATCH = True  # set to False to run just one file


def run_backprojection(FILE):

    # Loading files
    with open(FILE, "rb") as f:
        data = pkl.load(f)

    """
    Getting Data from the file and converting to a torch tensor. Using ascontiguousarray 
    to make sure all data in array is in one block of memory.
    This allows us to get more optimized performance 
    when indeixing through the array and performing operations. 
    """

    range_bins = tc.from_numpy(np.ascontiguousarray(data["range_bins"])).to(
        DEVICE, tc.float32
    )
    raw_scans = tc.from_numpy(np.ascontiguousarray(data["scan_data"])).to(
        DEVICE, tc.complex64
    )
    platform_pos = tc.from_numpy(np.ascontiguousarray(data["platform_pos"])).to(
        DEVICE, tc.float32
    )

    def retrieve_complex_data(real_data, range_bins):
        transform = tc.fft.fft(real_data, range_bins.shape[0])
        midIndex = transform.shape[1] // 2
        H = tc.zeros(transform.shape[1], device=transform.device, dtype=transform.dtype)
        H[0] = 1
        H[midIndex] = 1
        H[1:midIndex] = 2
        transform *= H
        transform = tc.fft.ifft(transform, axis=-1)

        return transform * tc.exp(-4j * tc.pi * range_bins[None, :] * FC / C)

    scan_data = retrieve_complex_data(raw_scans, range_bins)
    num_pulses, num_bins = scan_data.shape

    TERRAIN_SIZE = 3.0  # meters
    TERRAIN_RES = 0.01  # 1 cm per pixel

    x_center = 0.7  # meters, same as your cupy CENTER_X
    y_center = -0.15  # meters, same as your cupy CENTER_Y

    resolution = int(TERRAIN_SIZE / TERRAIN_RES)

    x_grid = tc.linspace(
        x_center - TERRAIN_SIZE / 2,
        x_center + TERRAIN_SIZE / 2,
        resolution,
        device=DEVICE,
    )
    y_grid = tc.linspace(
        y_center - TERRAIN_SIZE / 2,
        y_center + TERRAIN_SIZE / 2,
        resolution,
        device=DEVICE,
    )

    Xg, Yg = tc.meshgrid(x_grid, y_grid, indexing="xy")
    Zg = tc.zeros_like(Xg)
    grid = tc.stack([Xg, Yg, Zg], dim=-1)

    # 4) backproject with your existing pulse‐by‐pulse loop:
    img = tc.zeros(resolution, resolution, dtype=tc.complex64, device=DEVICE)
    r0 = range_bins[0].item()
    dr = (range_bins[1] - r0).item()

    tic = time.time()
    for m in range(num_pulses):
        Pm = platform_pos[m]
        Rm = tc.norm(grid - Pm, dim=-1)
        idx = ((Rm - r0) / dr).clamp(0, num_bins - 1.001)
        iL = idx.long()
        iH = iL + 1
        w = idx - iL.float()
        phase2 = tc.exp(4j * tc.pi * FC / C * Rm)
        A = scan_data[m, iL] + w * (scan_data[m, iH] - scan_data[m, iL])
        img += A * phase2

    print("BP finished in", time.time() - tic, "s")

    # 6) to dB + filter + plot
    img_db = 20 * tc.log10(img.abs().clamp_min(1e-12))
    image_db = img_db.cpu().numpy()
    smoothed = gaussian_filter(image_db, sigma=1.5)
    filtered = np.array(Filter.enhanced_lee(smoothed, 4), np.float32)
    db_max = filtered.max()
    db_min = filtered.min()

    # data‑driven color stretch
    vmin = db_max - 15
    vmax = db_max - 1

    return filtered, vmin, vmax, x_grid, y_grid

if SHOW_BATCH:
    files = sorted(glob.glob(os.path.join(BATCH_DIR, "*.pkl")))
    n = len(files)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    # Handle single subplot case
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, file in zip(axes, files):
        try:
            img, vmin, vmax, x_grid, y_grid = run_backprojection(file)

            # Grid boundaries for extent
            xmin = float(x_grid[0].cpu())
            xmax = float(x_grid[-1].cpu())
            ymin = float(y_grid[0].cpu())
            ymax = float(y_grid[-1].cpu())

            ax.imshow(
                img,
                extent=[xmin, xmax, ymin, ymax],
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            ax.set_title(os.path.basename(file))
            ax.set_xlabel("Cross Range (m)")
            ax.set_ylabel("Range (m)")
            ax.grid(True, linestyle="--", alpha=0.5)

        except Exception as e:
            print(f"Error with file {file}: {e}")
            ax.set_visible(False)

    # Hide any unused axes (in case total files < rows*cols)
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

else:
    # Run single file normally
    img, vmin, vmax, x_grid, y_grid = run_backprojection(FILE)
    plt.figure(figsize=(8, 6))
    xmin = float(x_grid[0].cpu().item())
    xmax = float(x_grid[-1].cpu().item())
    ymin = float(y_grid[0].cpu().item())
    ymax = float(y_grid[-1].cpu().item())

    plt.imshow(
        img,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    plt.colorbar(label="Amplitude (dB)")
    plt.title("SAR Backprojection")
    plt.xlabel("Cross Range (m)")
    plt.ylabel("Range (m)")
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.4, linestyle=":")
    plt.show()
