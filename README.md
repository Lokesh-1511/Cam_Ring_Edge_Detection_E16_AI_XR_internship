# Cam Ring Edge Detection and Dimensional Analysis

Professional computer vision project for AI/XR internship submission. The system analyzes top-view cam ring images, detects inner and outer edges, and computes thickness as a function of angle for dimensional inspection.

## Key Features

- Detects inner and outer cam ring boundaries from noisy images.
- Computes `thickness(θ) = outer_radius(θ) - inner_radius(θ)` at regular angular intervals.
- Supports batch processing for common image formats (`jpg`, `jpeg`, `png`, `bmp`, `tif`, `tiff`).
- Uses OpenCV + NumPy based preprocessing and edge extraction pipeline.
- Includes robustness methods for real-world defects/noise:
  - adaptive preprocessing options
  - radial band filtering for edge classification
  - angular continuity constraints
  - optional spike suppression on thickness profile
- Exports engineering-ready outputs (overlay image, plot, CSV).

## Workflow Overview

1. Load one or more cam ring images from an input folder.
2. Preprocess image (grayscale, Gaussian blur, optional adaptive threshold).
3. Detect edges using Canny (auto-tuned or manual thresholds).
4. Estimate ring center (Hough circle with sanity fallback).
5. Convert edge pixels to polar coordinates around the detected center.
6. Extract inner/outer radii per angle with radial-band constraints.
7. Apply continuity and optional de-spiking for stable profiles.
8. Compute thickness vs angle and save visual + numerical outputs.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Basic batch run:

```bash
python cam_ring_analysis.py --input-dir . --output-dir outputs
```

Single file pattern:

```bash
python cam_ring_analysis.py --input-dir . --output-dir outputs --pattern "CamRing.jpg"
```

Robust configuration for reflective/noisy parts:

```bash
python cam_ring_analysis.py \
  --input-dir . \
  --output-dir outputs \
  --angle-step 1 \
  --despike \
  --inner-min-ratio 0.15 \
  --inner-max-ratio 0.55 \
  --outer-min-ratio 0.55 \
  --outer-max-ratio 0.95 \
  --max-radius-jump 40
```

Manual Canny thresholds:

```bash
python cam_ring_analysis.py --input-dir . --canny-low 60 --canny-high 180
```

## Output Artifacts

For each processed image, the pipeline saves:

- `*_processed.png`: overlay with detected edges, center, boundaries, and radial sampling lines.
- `*_thickness_plot.png`: angle vs thickness graph.
- `*_thickness.csv`: numerical results with columns:
  - `angle_deg`
  - `inner_radius`
  - `outer_radius`
  - `thickness`

## Technologies Used

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- SciPy

## Future Improvements

- Pixel-to-millimeter calibration with reference scale.
- Automated parameter optimization across image batches.
- Advanced contour validation for partial occlusion cases.
- Optional report generation for quality-inspection workflows.

## Author

- Name: `<Your Name>`
- Role: AI/XR Internship Candidate
