# Team 4 BWSI SAR 2025 DEMO Code 


HELLO! This is not the original Repository; the original repository is private and is not controlled by me. However, as it was our team's code, I have transferred part of our code to demonstrate the level of coding that we have done. I will include the commit histories from the original repository in the media within the portfolio.

In order to make it easy, I have put together a demo version that can run the main parts of our code consistently. Please note that this is stemming from an older commit, and new commits were made to improve our code, such as GUIs. In the past, we had introduced Gradient Ascent algorithms to detect peaks, but we have removed that as it was more computationally heavy and proved unnecessary. 


This repository contains code for Team 4's Synthetic Aperture Radar (SAR) project for BWSI 2025. 

This system successfully generates detailed images when conducting indoor synthetic aperture scans from a TDSR Pulson452 system mounted on a hexcopter, with positioning data generated from an Optitrack Motion Capture system. Individual soft drink cans can be easily resolved when placed 40-50cm apart. 

The system has a variety of advanced features, including quaternion boresight positioning during alignment, oversampling protection during backprojection, and Hilbert transform complex data extraction.  


To run the demo, I have set up the following files:

- RTI_plotter.py - Plots a Range-Time-Intensity Plot, which is used to identify strong reflective objects over time.

- alignment.py - AUTOMATICALLY aligns the Range-time-intensity data received from the Radar with the Motion Capture system data to create a pickle file. 

- npBackProjectBlob.py - Uses the alignment_data.pkl file from data directories to create a 2 Map with Intensity Color map.


To be able to run these, I have included data collected from 4 of our trials, which are the four Aug1 directories at the top. The Batch directory and aligned_data.pkl within Flight 1 and 4 are created upon running alignment.py on a chosen directory. To allow you to test this, I have left Flight2 and Flight3 alone. 

- To choose which directory to run for alignment.py, please look for line 21 in the file.

- To run rti_plotter, you can choose any of the 4 directories. Please look for line 170 in the file to change directories.

- To run npBackProjectBlob.py, there must be both the Batch Directory and aligned_data.pkl. Please navigate to line 61 in the file to change directories.



(Feel free to read the rest of our readme as you wish :)


## Project Structure

```
pi/
    main.py
    datalogger.py
    pulson/
        config.py
        pulsonAPI.py
        singleScan.py
helpers/
    futuramediumbt.ttf
    ssh_gui.py
postprocessing/
    backprojectionOptions/
        cupyBackProjectBlob.py
        Filter.py
        globalOptimization.py
        npBackProjectBlob.py
        Torch_Backprojection.py
    3D-plot.py
    RTI_Plot.py
    alignment.py
    QuaternionKalmanFilter.py
    RTI_man_alignment.py
readme.md
```

## Components

- **pi/main.py**: Entry point for running the SAR data acquisition and communication with the Pulson device.
- **pi/datalogger.py**: Datalogging class for handling recording radar data and config info, recorded in binary and json respectively.
- **pi/pulson/pulsonAPI.py**: Main API for communicating with the Pulson radar hardware over UDP.
- **pi/pulson/singleScan.py**: Class for handling a single radar scan's data and converting time to range.
- **postprocessing/RTI_Plot.py**: Utilities for plotting Range-Time-Intensity (RTI) images using matplotlib.
- **postprocessing/alignment.py**: Automatic position/radar data alignment function
- **postprocessing/RTI_man_alignment.py**: Manual position/radar data alignment function
- **postprocessing/backprojectionOptions**: Various backprojection algorithms for different hardware and their helper functions.
- **helpers/ssh_gui.py** Easy to use SSH GUI for configuring radar remotely.

## Usage

1. **Install dependencies**  
   Make sure you have Python 3.13 and the following packages or better versions:
   - numpy 2.2.6
   - numpy-quaternion 2024.0.10
   - matplotlib 3.10.3
   - scikit-learn 1.7.0
   - scikit-image 0.25.2
   - OpenCV 4.12.0.88
   - PyWavelets 1.8.0
   - Pandas 2.3.1
   - PySide6 6.9.1
   - (Optional) cupy 13.5.1 & CUDA Toolkit v12.x
   - (Optional) PyTorch 2.7.1

   You can install them with:
   ```sh
   pip install numpy matplotlib pandas scikit-learn scikit-image cupy-cuda12x torch numpy-quaternion PyWavelets PySide6 opencv-python

   ```

2. **Run the Radar Program**  
   Move the entire pi directory to your Raspberry Pi host, connected to the Pulson through UDP. You can achieve this through SCP or a button in `ssh_gui.py`
   
   Edit the IP address in `pi/main.py` if needed, then run:
   ```sh
   python pi/main.py
   ```
   Data will be recorded in a new directory under DATAS named the current timestamp, with `metadata.json` and `returns.bin` files.
   
   Running the program remotely can be accomplished with `helpers/ssh_gui`. The GUI will connect to the pi via SSH address, send and confirm configuration, and fire/stop the radar. Data is saved locally to the raspberry pi, and if avaliable, to a mounted USB drive, with the directory name of the current pi datetime. 

   For best effects, activate the Motion Capture system about 2-3 seconds before firing the radar, and end it 2-3 seconds after stopping the radar.  

3. **Plotting RTI**  
   Use `postprocessing/RTI_Plot.py` to visualize RTI data arrays.
   
   Change filepath in `RTI_Plot.py` to directory of interest. Ensure the directory has both the `metadata.json` and `returns.bin` files.

4. **Aligning Data**  
   Use `postprocessing/alignment.py` to automatically align RTI and motion capture data.
   
   The alignment relies on Optitrack Motion Capture data. In the Motive software, ensure that the imaging platform is defined as a rigidbody, and that the export track contains headers, rigidbody coordinate positions, and rigidbody quaternion orientation data. Also make note of known reflector locations. Export the track in the form of a csv file with the first body being the platform rigidbody, and name it `pos.csv`
   
   Change filepath in `alignment.py` to directory of interest. Ensure the directory has `metadata.json`, `returns.bin`, and `pos.csv` files. Change the `actual_square0_pos` and `actual_square1_pos` positions to known reflector locations in Motion Capture coordinate frame. 
   
   `alignment.py` will generate a `.pkl` file in the directory of interest with aligned data in and visualize the alignment on an RTI plot. 
   
   If the `BATCH` parameter is enabled, the script will further generate `BATCH_COUNT` number of `.pkl` files that apply a `BATCH_TIME_STEP` offset evenly spaced about the original alignment time, in a subdirectory named `BATCH`. This is especially helpful because from experience, an alignment offset of 40ms can make or break a backprojection. The current settings reliably generate at least one good backprojection image, but you may wish to tune the number or density of alignment batches.  
   
   If additional tuning is desired, `DX_SEARCH` and `DT_SEARCH` parameters can be adjusted. 

5. **Backprojection**  
   Pick a backprojection method of choice in the `postprocessing/backprojectionOptions` directory. Set the directory in each file to the `.pkl` file of interest and confirm correct image center, resolution, and size params.
   - `npBackProjectBlob.py` is a general purpose, multithreaded, numpy based BP algorithm that can be used on any system, including auto dynamic range through Laplacian of Gaussian reflector detection.
   - `cupyBackProjectBlob.py` is a CUDA accelerated version of `npBackProjectBlob`. It is much more performant if your system has a CUDA capable GPU. Ensure you have cupy installed. This script also enables oversampling protection and batch run capability, enabled by `NORMALIZE_DISTANCE` and `BATCH_RUN` parameters respectively. Should `BATCH_RUN` be specified, provide a directory with the `BATCH` folder instead of the specific `.pkl` file. 
   - `Torch_Backprojection.py` is a seperate Torch backprojection algorithm. The feature set is slightly different and does not include auto dynamic range. However, Torch runs well on Apple Silicon hardware. 

   Further, `Filter.py` contain some interesting filters that can be applied simply by calling the `filter` library in any backprojection algorithm and providing it with the numpy array of the backprojected image. We've seen best success with the Anisotropic filter. Specific documentation is avaliable in `Filter.py`.


## Notes

- Communication with the Pulson device is handled via UDP sockets.
- Modify `pi/pulson/config.py` for configuration options. Do not change defaults unless needed

## Authors

Team 4, BWSI
