# LiDAR-Camera Fusion for 3D Object Localization & Signal Processing Explorations

## Project Overview

This project demonstrates a pipeline for achieving 3D object localization by fusing data from a vehicle-mounted camera and a LiDAR sensor, using the KITTI raw dataset. The primary approach involves performing 2D object detection on camera images and then associating these detections with LiDAR point cloud data to estimate depth and subsequently their 3D world coordinates.

The project also serves as an exploration of various "Computing with Signals" concepts, applying signal processing techniques to the image, LiDAR, and IMU/GPS signals involved.

## Core Pipeline

1.  **Data Acquisition & Calibration:**

    - Uses KITTI raw data sequences (synchronized camera images, Velodyne LiDAR scans, OXTS (IMU/GPS) data).
    - Loads and utilizes calibration files (`calib_cam_to_cam.txt`, `calib_velo_to_cam.txt`, `calib_imu_to_velo.txt`) to establish geometric relationships between sensor coordinate frames.
    - Calculates transformation matrices (e.g., `T_velo_cam2`, `T_cam2_imu`) for projecting points between LiDAR, camera, and IMU frames.

2.  **2D Object Detection:**

    - Employs the YOLOv5 model (loaded from PyTorch Hub) to detect objects in the 2D camera images (image signal processing).

3.  **Depth Association & 3D Localization:**

    - LiDAR point clouds (3D spatial signal) are loaded, and the ground plane is removed using the RANSAC algorithm (robust model fitting on noisy signal data).
    - Remaining LiDAR points are projected into the camera's image plane.
    - The 2D bounding box center of each detected object is associated with the nearest projected LiDAR point to estimate its depth (z-coordinate in the camera frame).
    - These (u,v,z) camera coordinates are then transformed into the IMU's (x,y,z) coordinate system and subsequently to global Latitude, Longitude, and Altitude (LLA) coordinates using the vehicle's OXTS data.

4.  **Initial Visualization:**
    - A video (`lidar_stack.mp4`) is generated, stacking:
      - The camera view with 2D detections and depth estimates overlaid.
      - The projected LiDAR point cloud (color-coded by depth).
      - A simple top-down scenario view showing the ego vehicle and detected objects in the IMU frame.

## Elaborations & Signal Processing Explorations

The notebook includes several cells that build upon the core pipeline, demonstrating various signal processing concepts:

1.  **Quantitative Evaluation (Conceptual):**

    - Discusses the need for metrics like 3D Intersection over Union (IoU) and Average Precision (AP) for rigorous performance assessment, which would require 3D ground truth labels from the standard KITTI detection dataset.

2.  **Bird's-Eye View (BEV) Visualization:**

    - Generates a top-down BEV map from the raw LiDAR point cloud.
    - Points are colored by height to visualize the 3D structure.
    - Detected object centers (transformed to the LiDAR frame) are overlaid.
    - This provides an alternative spatial representation of the 3D signal, aiding in qualitative analysis.
    - _(Output: `bev_map_frame_10.png` displayed in notebook, also offered for download)._

3.  **Refined Depth Association:**

    - Implements `get_uvz_centers_refined` which, instead of using the single nearest LiDAR point, averages the depth of multiple LiDAR points falling within an object's 2D bounding box.
    - This acts as a local signal averaging technique to produce more robust depth estimates by reducing noise.
    - _(Output: `lidar_stack_refined.mp4` video for comparison)._

4.  **LiDAR Signal Filtering (Voxel Downsampling) & Performance Analysis:**

    - Demonstrates voxel downsampling on the LiDAR point cloud using the `voxel_downsample` function. This technique reduces data density by representing points within each voxel by their centroid, acting as a spatial low-pass filter.
    - Compares the time taken to generate a BEV map using raw vs. downsampled points, quantifying the processing speedup.
    - Visually compares the BEV maps from raw and downsampled data.
    - _(Output: Timing comparison printed, `bev_comparison_frame_10.png` displayed in notebook, also offered for download)._

5.  **Object Tracking with Kalman Filter:**
    - Implements a simple 2D Kalman Filter (`KalmanFilter2D` class) to track detected objects in the IMU frame and smooth their trajectories over time.
    - The Kalman filter performs state estimation (position and velocity) from noisy measurements (frame-by-frame detections) using a predict-update cycle. This is a classic time-series signal processing technique for noise reduction and trajectory smoothing.
    - _(Output: `lidar_stack_kalman.mp4` video showing smoother object tracks)._

## Requirements

- Python 3
- OpenCV (`cv2`)
- NumPy
- Pandas
- Matplotlib
- PyTorch & YOLOv5 (`torch.hub.load('ultralytics/yolov5', 'yolov5s')`)
- `pymap3d`
- `IPython` (for display in Colab)
- `kitti_utils.py` (provided in the notebook via `wget`)

## How to Run

The project is structured as a Google Colaboratory notebook (`finalsignal.py`). You can access the notebook here [Colab Link](https://colab.research.google.com/drive/1NSjc0a9n0_O78aM25dZ0zMVZpeO8GHeV?usp=sharing).

1.  **Open in Colab:** Upload and open `finalsignal.py` in Google Colab.
2.  **Run Setup Cells:**
    - Execute the initial cells to download KITTI raw data (`2011_10_03_drive_0047_sync.zip` and `2011_10_03_calib.zip`) and extract it.
    - Run cells for library imports and `wget` for `kitti_utils.py`.
    - Run cells that load calibration data and compute transformation matrices.
    - Run the cell to load the YOLOv5 model.
3.  **Original Pipeline & Video:**
    - Execute cells defining `get_uvz_centers`, `get_detection_coordinates`, `imu2geodetic`, `draw_scenario`.
    - Run the single frame test cell.
    - Run the video generation loop that creates and displays/downloads `lidar_stack.mp4`. Ensure the ego-vehicle coordinate definitions are placed _before_ this loop for correct visualization.
4.  **Elaborations:**
    - **BEV Visualization:** Run the cell under "Elaboration: Bird's-Eye View Visualization" to see the BEV map for frame 10.
    - **Refined Depth:** Run the cell under "Running Modified Pipeline with Refined Depth Association" to generate `lidar_stack_refined.mp4`.
    - **Voxel Downsampling & Timing:** Run the cell under "Elaboration: Demonstrating Time Benefit, Saving & Displaying BEV Comparison" to see the timing impact and the comparison plot (`bev_comparison_frame_10.png`).
    - **Kalman Filter Tracking:** Run the cell under "Elaboration: Object Tracking with Kalman Filter" to generate `lidar_stack_kalman.mp4`.
5.  **Downloads (Optional):**
    - Cells at the end provide options to download the generated videos and BEV map images using `google.colab.files.download()`.

## Outputs

- `lidar_stack.mp4`: Video from the original pipeline.
- `lidar_stack_refined.mp4`: Video from the pipeline with refined depth association.
- `lidar_stack_kalman.mp4`: Video from the pipeline with Kalman filter tracking.
- `bev_map_frame_10.png` (displayed/downloadable): Bird's-Eye View of frame 10.
- `bev_comparison_frame_10.png` (displayed/downloadable): Side-by-side BEV of raw vs. downsampled LiDAR for frame 10.
- Printouts of timing comparisons for voxel downsampling.
- Various plots and image displays throughout the notebook.

## Key Functions (from `kitti_utils.py` and notebook)

- `get_oxts()`: Loads IMU/GPS data.
- `bin2xyzw()` (from `kitti_utils.py`, assumed name based on context): Loads and processes raw LiDAR `.bin` files.
- `project_velobin2uvz()`: Projects LiDAR points to the camera image plane, including RANSAC ground removal.
- `get_rigid_transformation()`: Reads calibration files to get transformation matrices.
- `transform_uvz()`: Transforms (u,v,z) camera coordinates to another frame (e.g., IMU).
- `get_uvz_centers()` / `get_uvz_centers_refined()`: Associates 2D detections with LiDAR depth.
- `get_detection_coordinates()` / `get_detection_coordinates_refined()` / `get_detection_coordinates_for_kf()`: Main functions orchestrating detection and coordinate estimation for each frame.
- `imu2geodetic()`: Converts IMU (x,y,z) to LLA.
- `draw_velo_on_image()`: Visualizes projected LiDAR points on an image.
- `draw_scenario()` / `draw_scenario_tracked_kalman()`: Creates the top-down schematic view.
- `create_bev_map()` / `draw_ego_on_bev()` / `draw_boxes_on_bev()`: Functions for generating the Bird's-Eye View map.
- `voxel_downsample()`: Performs voxel grid downsampling on point clouds.
- `KalmanFilter2D`: Class implementing the Kalman filter for tracking.

---

_This README was generated based on the content of `finalsignal.py`._
