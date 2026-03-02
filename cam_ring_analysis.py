import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image(image_path: Path) -> np.ndarray:
    """Load an image from disk.

    Args:
        image_path: Path to the image file.

    Returns:
        Loaded BGR image as NumPy array.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If OpenCV cannot decode image.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image (unsupported/corrupt): {image_path}")

    return image


def preprocess(
    image_bgr: np.ndarray,
    blur_ksize: int = 5,
    use_adaptive_thresh: bool = False,
    adaptive_block_size: int = 31,
    adaptive_c: int = 2,
) -> Dict[str, np.ndarray]:
    """Preprocess image with grayscale, blur, and optional adaptive thresholding."""
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    thresholded = None
    if use_adaptive_thresh:
        if adaptive_block_size % 2 == 0:
            adaptive_block_size += 1
        thresholded = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            adaptive_block_size,
            adaptive_c,
        )

    return {"gray": gray, "blurred": blurred, "thresholded": thresholded}


def auto_canny_thresholds(image_gray: np.ndarray, sigma: float = 0.33) -> Tuple[int, int]:
    """Estimate Canny thresholds automatically from median intensity."""
    median_val = float(np.median(image_gray))
    low = int(max(0, (1.0 - sigma) * median_val))
    high = int(min(255, (1.0 + sigma) * median_val))
    if high <= low:
        high = min(255, low + 40)
    return low, high


def detect_edges(
    processed: Dict[str, np.ndarray],
    canny_low: Optional[int] = None,
    canny_high: Optional[int] = None,
    sigma: float = 0.33,
    use_thresholded_for_edges: bool = False,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Detect edges using Canny with manual or automatically tuned thresholds."""
    source = processed["thresholded"] if use_thresholded_for_edges and processed["thresholded"] is not None else processed["blurred"]

    if canny_low is None or canny_high is None:
        low, high = auto_canny_thresholds(source, sigma=sigma)
    else:
        low, high = int(canny_low), int(canny_high)

    edges = cv2.Canny(source, low, high)
    return edges, (low, high)


def find_center(
    processed: Dict[str, np.ndarray],
    edges: np.ndarray,
    min_radius_ratio: float = 0.1,
    max_radius_ratio: float = 0.95,
    sanity_offset_ratio: float = 0.25,
) -> Tuple[Tuple[int, int], str]:
    """Find cam ring center via Hough circles, fallback to contour centroid."""
    gray = processed["gray"]
    height, width = gray.shape[:2]
    min_dim = min(height, width)

    min_radius = max(5, int(min_dim * min_radius_ratio))
    max_radius = max(min_radius + 5, int(min_dim * max_radius_ratio / 2))

    contour_centroid: Optional[Tuple[int, int]] = None
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:
            contour_centroid = (
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]),
            )

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dim / 4,
        param1=120,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None and len(circles) > 0:
        circles_rounded = np.round(circles[0]).astype(int)
        best_circle = max(circles_rounded, key=lambda c: c[2])
        cx, cy = int(best_circle[0]), int(best_circle[1])

        img_center = np.array([width / 2.0, height / 2.0], dtype=float)
        det_center = np.array([cx, cy], dtype=float)
        offset = float(np.linalg.norm(det_center - img_center))

        if offset > sanity_offset_ratio * min_dim and contour_centroid is not None:
            print("WARNING: Hough center suspicious, using contour centroid fallback")
            return contour_centroid, "contour_centroid_sanity_fallback"

        return (cx, cy), "hough"

    if contour_centroid is not None:
        return contour_centroid, "contour_centroid"

    return (width // 2, height // 2), "image_center_fallback"


def _interpolate_missing(values: np.ndarray) -> np.ndarray:
    """Interpolate NaN values circularly if enough valid values exist."""
    out = values.astype(float).copy()
    n = len(out)
    valid = np.isfinite(out)

    if valid.sum() < 2:
        return out

    x = np.arange(n)
    valid_x = x[valid]
    valid_y = out[valid]

    extended_x = np.concatenate([valid_x - n, valid_x, valid_x + n])
    extended_y = np.concatenate([valid_y, valid_y, valid_y])

    missing = ~valid
    out[missing] = np.interp(x[missing], extended_x, extended_y)
    return out


def extract_radii(
    edges: np.ndarray,
    center: Tuple[int, int],
    angle_step: float = 1.0,
    min_points_per_bin: int = 1,
    global_clip_percentiles: Tuple[float, float] = (1.0, 99.0),
    inner_percentile: float = 10.0,
    outer_percentile: float = 90.0,
    inner_min_ratio: float = 0.15,
    inner_max_ratio: float = 0.55,
    outer_min_ratio: float = 0.55,
    outer_max_ratio: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract inner/outer radii per angle from edge pixels in polar coordinates."""
    if angle_step <= 0 or angle_step > 180:
        raise ValueError("angle_step must be in (0, 180].")

    cx, cy = center
    ys, xs = np.where(edges > 0)

    if len(xs) == 0:
        raise RuntimeError("No edge pixels detected. Try different preprocessing/Canny parameters.")

    dx = xs.astype(float) - float(cx)
    dy = ys.astype(float) - float(cy)
    radii = np.hypot(dx, dy)
    angles_deg = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

    if not (0.0 <= inner_min_ratio < inner_max_ratio <= 1.0):
        raise ValueError("Inner radius ratios must satisfy 0 <= min < max <= 1.")
    if not (0.0 <= outer_min_ratio < outer_max_ratio <= 1.0):
        raise ValueError("Outer radius ratios must satisfy 0 <= min < max <= 1.")
    if inner_max_ratio > outer_min_ratio:
        raise ValueError("inner_max_ratio should be <= outer_min_ratio to separate bands.")

    corner_distances = np.array(
        [
            np.hypot(float(cx), float(cy)),
            np.hypot(float(cx), float(edges.shape[0] - 1 - cy)),
            np.hypot(float(edges.shape[1] - 1 - cx), float(cy)),
            np.hypot(float(edges.shape[1] - 1 - cx), float(edges.shape[0] - 1 - cy)),
        ],
        dtype=float,
    )
    max_possible_radius = float(np.max(corner_distances))
    if max_possible_radius <= 0:
        raise RuntimeError("Invalid center for radius normalization.")

    clip_lo = float(np.percentile(radii, global_clip_percentiles[0]))
    clip_hi = float(np.percentile(radii, global_clip_percentiles[1]))
    keep = (radii >= clip_lo) & (radii <= clip_hi)
    radii = radii[keep]
    angles_deg = angles_deg[keep]
    radii_norm = radii / max_possible_radius

    sampled_angles = np.arange(0.0, 360.0, angle_step)
    n_bins = len(sampled_angles)

    inner = np.full(n_bins, np.nan, dtype=float)
    outer = np.full(n_bins, np.nan, dtype=float)

    bin_indices = np.floor(angles_deg / angle_step).astype(int) % n_bins

    for bin_idx in range(n_bins):
        in_bin = radii[bin_indices == bin_idx]
        in_bin_norm = radii_norm[bin_indices == bin_idx]
        if in_bin.size >= min_points_per_bin:
            inner_candidates = in_bin[(in_bin_norm > inner_min_ratio) & (in_bin_norm < inner_max_ratio)]
            outer_candidates = in_bin[(in_bin_norm > outer_min_ratio) & (in_bin_norm < outer_max_ratio)]

            if inner_candidates.size >= min_points_per_bin:
                inner[bin_idx] = float(np.percentile(inner_candidates, inner_percentile))
            if outer_candidates.size >= min_points_per_bin:
                outer[bin_idx] = float(np.percentile(outer_candidates, outer_percentile))

    inner = _interpolate_missing(inner)
    outer = _interpolate_missing(outer)

    invalid = np.isfinite(inner) & np.isfinite(outer) & (outer < inner)
    if np.any(invalid):
        swapped_inner = inner[invalid].copy()
        inner[invalid] = outer[invalid]
        outer[invalid] = swapped_inner

    return sampled_angles, inner, outer


def compute_thickness(inner_radius: np.ndarray, outer_radius: np.ndarray) -> np.ndarray:
    """Compute thickness = outer - inner, preserving NaN for invalid entries."""
    thickness = outer_radius - inner_radius
    thickness[~(np.isfinite(inner_radius) & np.isfinite(outer_radius))] = np.nan
    return thickness


def smooth_radius(radius_values: np.ndarray, max_jump: float = 40.0) -> np.ndarray:
    """Enforce angular continuity by limiting step-to-step radius jumps."""
    smoothed = radius_values.astype(float).copy()
    if max_jump <= 0:
        return smoothed

    for idx in range(1, len(smoothed)):
        if not (np.isfinite(smoothed[idx]) and np.isfinite(smoothed[idx - 1])):
            continue
        if abs(smoothed[idx] - smoothed[idx - 1]) > max_jump:
            smoothed[idx] = smoothed[idx - 1]

    if len(smoothed) > 1 and np.isfinite(smoothed[0]) and np.isfinite(smoothed[-1]):
        if abs(smoothed[0] - smoothed[-1]) > max_jump:
            smoothed[0] = smoothed[-1]

    return smoothed


def adaptive_radius_jump(
    outer_radius: np.ndarray,
    jump_scale: float = 0.05,
    fallback_jump: float = 40.0,
) -> float:
    """Compute adaptive radius jump limit from outer radius mean."""
    finite_outer = outer_radius[np.isfinite(outer_radius)]
    if finite_outer.size == 0:
        return fallback_jump

    adaptive_jump = jump_scale * float(np.nanmean(finite_outer))
    if not np.isfinite(adaptive_jump) or adaptive_jump <= 0:
        return fallback_jump
    return adaptive_jump


def _circular_median(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Compute circular rolling median for 1D values with odd-sized window."""
    if window < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1

    pad = window // 2
    wrapped = np.concatenate([values[-pad:], values, values[:pad]])
    out = np.full_like(values, np.nan, dtype=float)

    for i in range(len(values)):
        segment = wrapped[i : i + window]
        finite = segment[np.isfinite(segment)]
        if finite.size > 0:
            out[i] = float(np.median(finite))

    return out


def despike_thickness(
    thickness: np.ndarray,
    window: int = 7,
    mad_thresh: float = 4.0,
) -> np.ndarray:
    """Suppress spike outliers using circular median baseline and MAD threshold."""
    out = thickness.astype(float).copy()
    valid = np.isfinite(out)
    if valid.sum() < max(5, window):
        return out

    baseline = _circular_median(out, window=window)
    residual = out - baseline
    finite_res = residual[np.isfinite(residual)]
    if finite_res.size < 5:
        return out

    mad = float(np.median(np.abs(finite_res - np.median(finite_res))))
    if mad <= 1e-9:
        return out

    robust_sigma = 1.4826 * mad
    spikes = np.isfinite(residual) & (np.abs(residual) > mad_thresh * robust_sigma)
    out[spikes] = baseline[spikes]
    return out


def _radii_to_points(
    center: Tuple[int, int],
    angles_deg: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Convert polar radii arrays to contour-like cartesian integer points."""
    cx, cy = center
    pts: List[List[int]] = []

    for angle, radius in zip(angles_deg, radii):
        if not np.isfinite(radius):
            continue
        theta = math.radians(float(angle))
        x = int(round(cx + radius * math.cos(theta)))
        y = int(round(cy + radius * math.sin(theta)))
        pts.append([x, y])

    if len(pts) < 2:
        return np.empty((0, 1, 2), dtype=np.int32)

    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def _save_csv(
    csv_path: Path,
    angles_deg: np.ndarray,
    inner_radius: np.ndarray,
    outer_radius: np.ndarray,
    thickness: np.ndarray,
    pixel_to_mm: Optional[float] = None,
) -> None:
    """Save angular radii/thickness data to CSV."""
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if pixel_to_mm is None:
            writer.writerow(["angle_deg", "inner_radius", "outer_radius", "thickness"])
            for angle, inner, outer, thick in zip(angles_deg, inner_radius, outer_radius, thickness):
                writer.writerow([f"{angle:.3f}", f"{inner:.6f}", f"{outer:.6f}", f"{thick:.6f}"])
        else:
            writer.writerow(
                [
                    "angle_deg",
                    "inner_radius",
                    "outer_radius",
                    "thickness",
                    "inner_radius_mm",
                    "outer_radius_mm",
                    "thickness_mm",
                ]
            )
            for angle, inner, outer, thick in zip(angles_deg, inner_radius, outer_radius, thickness):
                inner_mm = inner * pixel_to_mm
                outer_mm = outer * pixel_to_mm
                thick_mm = thick * pixel_to_mm
                writer.writerow(
                    [
                        f"{angle:.3f}",
                        f"{inner:.6f}",
                        f"{outer:.6f}",
                        f"{thick:.6f}",
                        f"{inner_mm:.6f}",
                        f"{outer_mm:.6f}",
                        f"{thick_mm:.6f}",
                    ]
                )


def visualize_results(
    image_bgr: np.ndarray,
    edges: np.ndarray,
    center: Tuple[int, int],
    angles_deg: np.ndarray,
    inner_radius: np.ndarray,
    outer_radius: np.ndarray,
    thickness: np.ndarray,
    output_prefix: Path,
    show: bool = False,
    radial_line_step: int = 10,
    pixel_to_mm: Optional[float] = None,
) -> Dict[str, Path]:
    """Create and save overlay visualization and thickness plot."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    overlay = image_bgr.copy()

    edge_layer = np.zeros_like(overlay)
    edge_layer[edges > 0] = (255, 255, 255)
    overlay = cv2.addWeighted(overlay, 0.85, edge_layer, 0.65, 0)

    inner_pts = _radii_to_points(center, angles_deg, inner_radius)
    outer_pts = _radii_to_points(center, angles_deg, outer_radius)

    if len(inner_pts) > 1:
        cv2.polylines(overlay, [inner_pts], isClosed=True, color=(0, 255, 0), thickness=2)
    if len(outer_pts) > 1:
        cv2.polylines(overlay, [outer_pts], isClosed=True, color=(0, 0, 255), thickness=2)

    cx, cy = center
    cv2.circle(overlay, (cx, cy), 5, (0, 255, 255), -1)
    cv2.putText(
        overlay,
        "center",
        (cx + 8, cy - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if radial_line_step < 1:
        radial_line_step = 1

    indices = np.arange(0, len(angles_deg), radial_line_step)
    for idx in indices:
        if not (np.isfinite(inner_radius[idx]) and np.isfinite(outer_radius[idx])):
            continue
        theta = math.radians(float(angles_deg[idx]))
        x1 = int(round(cx + inner_radius[idx] * math.cos(theta)))
        y1 = int(round(cy + inner_radius[idx] * math.sin(theta)))
        x2 = int(round(cx + outer_radius[idx] * math.cos(theta)))
        y2 = int(round(cy + outer_radius[idx] * math.sin(theta)))
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 1)

    finite_thickness = np.where(np.isfinite(thickness))[0]
    if finite_thickness.size > 0:
        min_idx = int(finite_thickness[np.argmin(thickness[finite_thickness])])
        max_idx = int(finite_thickness[np.argmax(thickness[finite_thickness])])

        for idx, color, label in [
            (min_idx, (0, 165, 255), "min"),
            (max_idx, (255, 0, 255), "max"),
        ]:
            theta = math.radians(float(angles_deg[idx]))
            x1 = int(round(cx + inner_radius[idx] * math.cos(theta)))
            y1 = int(round(cy + inner_radius[idx] * math.sin(theta)))
            x2 = int(round(cx + outer_radius[idx] * math.cos(theta)))
            y2 = int(round(cy + outer_radius[idx] * math.sin(theta)))
            cv2.line(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                label,
                (x2 + 4, y2 + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    processed_image_path = output_prefix.with_name(f"{output_prefix.name}_processed.png")
    cv2.imwrite(str(processed_image_path), overlay)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if pixel_to_mm is None:
        thickness_plot_values = thickness
        thickness_label = "Thickness (pixels)"
    else:
        thickness_plot_values = thickness * pixel_to_mm
        thickness_label = "Thickness (mm)"

    ax.plot(angles_deg, thickness_plot_values, color="tab:blue", linewidth=1.5, label="Thickness")
    ax.set_title("Cam Ring Thickness vs Angle")
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel(thickness_label)
    ax.set_xlim(0, 360)
    ax.grid(True, alpha=0.3)

    if finite_thickness.size > 0:
        min_idx = int(finite_thickness[np.argmin(thickness_plot_values[finite_thickness])])
        max_idx = int(finite_thickness[np.argmax(thickness_plot_values[finite_thickness])])
        ax.scatter([angles_deg[min_idx]], [thickness_plot_values[min_idx]], color="orange", label="Min")
        ax.scatter([angles_deg[max_idx]], [thickness_plot_values[max_idx]], color="magenta", label="Max")

    ax.legend(loc="best")
    fig.tight_layout()

    plot_path = output_prefix.with_name(f"{output_prefix.name}_thickness_plot.png")
    fig.savefig(plot_path, dpi=180)

    if show:
        edge_preview = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Detected Edges", edge_preview)
        cv2.imshow("Cam Ring Analysis", overlay)
        plt.show(block=False)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        plt.close(fig)
    else:
        plt.close(fig)

    csv_path = output_prefix.with_name(f"{output_prefix.name}_thickness.csv")
    _save_csv(csv_path, angles_deg, inner_radius, outer_radius, thickness, pixel_to_mm=pixel_to_mm)

    return {
        "processed_image": processed_image_path,
        "thickness_plot": plot_path,
        "csv": csv_path,
    }


def _collect_input_images(input_dir: Path, pattern: Optional[str] = None) -> List[Path]:
    """Collect valid image files from directory using optional glob pattern."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist or is not a directory: {input_dir}")

    if pattern:
        candidates = sorted(input_dir.glob(pattern))
        return [p for p in candidates if p.suffix.lower() in VALID_EXTENSIONS and p.is_file()]

    images: List[Path] = []
    for ext in sorted(VALID_EXTENSIONS):
        images.extend(sorted(input_dir.glob(f"*{ext}")))
    return [p for p in images if p.is_file()]


def _print_summary(
    image_path: Path,
    center: Tuple[int, int],
    method: str,
    thresholds: Tuple[int, int],
    outputs: Dict[str, Path],
    thickness: np.ndarray,
    used_max_jump: float,
    pixel_to_mm: Optional[float] = None,
) -> None:
    finite = thickness[np.isfinite(thickness)]
    if finite.size:
        min_t = float(np.min(finite))
        max_t = float(np.max(finite))
        mean_t = float(np.mean(finite))
    else:
        min_t = max_t = mean_t = float("nan")

    print(f"\nProcessed: {image_path.name}")
    print(f"  Center: {center} (method={method})")
    print(f"  Canny thresholds: low={thresholds[0]}, high={thresholds[1]}")
    print(f"  Radius continuity max jump: {used_max_jump:.3f} px")
    print(f"  Thickness stats (pixels): min={min_t:.3f}, mean={mean_t:.3f}, max={max_t:.3f}")
    if pixel_to_mm is not None:
        print(
            "  Thickness stats (mm): "
            f"min={min_t * pixel_to_mm:.3f}, "
            f"mean={mean_t * pixel_to_mm:.3f}, "
            f"max={max_t * pixel_to_mm:.3f}"
        )
    print(f"  Outputs:")
    print(f"    - {outputs['processed_image']}")
    print(f"    - {outputs['thickness_plot']}")
    print(f"    - {outputs['csv']}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Edge detection and dimensional analysis of a cam ring."
    )

    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Folder containing input images.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Folder to save outputs.")
    parser.add_argument("--pattern", type=str, default=None, help="Optional glob pattern, e.g. '*.jpg'.")

    parser.add_argument("--angle-step", type=float, default=1.0, help="Angular sampling step in degrees.")
    parser.add_argument("--blur-ksize", type=int, default=5, help="Gaussian blur kernel size (odd).")
    parser.add_argument("--sigma", type=float, default=0.33, help="Auto-Canny sigma parameter.")
    parser.add_argument("--inner-min-ratio", type=float, default=0.15, help="Normalized min radius for inner edge band.")
    parser.add_argument("--inner-max-ratio", type=float, default=0.55, help="Normalized max radius for inner edge band.")
    parser.add_argument("--outer-min-ratio", type=float, default=0.55, help="Normalized min radius for outer edge band.")
    parser.add_argument("--outer-max-ratio", type=float, default=0.95, help="Normalized max radius for outer edge band.")
    parser.add_argument(
        "--max-radius-jump",
        type=float,
        default=None,
        help="Fixed max allowed radius jump between adjacent angles. If omitted, adaptive value is used.",
    )
    parser.add_argument(
        "--jump-scale",
        type=float,
        default=0.05,
        help="Adaptive jump scale factor: max_jump = jump_scale * mean(outer_radius).",
    )
    parser.add_argument(
        "--thickness-median-kernel",
        type=int,
        default=5,
        help="Median filter kernel size for thickness smoothing (0 to disable).",
    )
    parser.add_argument(
        "--pixel-to-mm",
        type=float,
        default=None,
        help="Optional scale factor to convert pixels to millimeters.",
    )

    parser.add_argument("--canny-low", type=int, default=None, help="Manual Canny low threshold.")
    parser.add_argument("--canny-high", type=int, default=None, help="Manual Canny high threshold.")

    parser.add_argument("--use-adaptive-thresh", action="store_true", help="Enable adaptive threshold preprocessing.")
    parser.add_argument(
        "--use-thresholded-for-edges",
        action="store_true",
        help="Run Canny on adaptive-threshold image when available.",
    )
    parser.add_argument(
        "--radial-line-step",
        type=int,
        default=10,
        help="Draw every Nth sampled radial thickness line on overlay.",
    )
    parser.add_argument(
        "--despike",
        action="store_true",
        help="Apply robust spike suppression to thickness curve.",
    )
    parser.add_argument(
        "--despike-window",
        type=int,
        default=7,
        help="Odd window size for circular rolling median in spike suppression.",
    )
    parser.add_argument(
        "--despike-mad-thresh",
        type=float,
        default=4.0,
        help="MAD-based threshold multiplier for spike suppression.",
    )
    parser.add_argument("--show", action="store_true", help="Show intermediate/final windows interactively.")

    return parser.parse_args()


def main() -> None:
    """Run batch cam ring analysis pipeline."""
    args = parse_args()

    try:
        image_paths = _collect_input_images(args.input_dir, args.pattern)
    except Exception as exc:
        print(f"Input error: {exc}")
        raise SystemExit(1) from exc

    if not image_paths:
        print("No input images found. Check --input-dir and --pattern.")
        raise SystemExit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    for image_path in image_paths:
        try:
            image_bgr = load_image(image_path)
            processed = preprocess(
                image_bgr,
                blur_ksize=args.blur_ksize,
                use_adaptive_thresh=args.use_adaptive_thresh,
            )

            edges, thresholds = detect_edges(
                processed,
                canny_low=args.canny_low,
                canny_high=args.canny_high,
                sigma=args.sigma,
                use_thresholded_for_edges=args.use_thresholded_for_edges,
            )

            center, method = find_center(processed, edges)
            angles_deg, inner_radius, outer_radius = extract_radii(
                edges,
                center,
                angle_step=args.angle_step,
                inner_min_ratio=args.inner_min_ratio,
                inner_max_ratio=args.inner_max_ratio,
                outer_min_ratio=args.outer_min_ratio,
                outer_max_ratio=args.outer_max_ratio,
            )

            used_max_jump = (
                float(args.max_radius_jump)
                if args.max_radius_jump is not None
                else adaptive_radius_jump(outer_radius, jump_scale=args.jump_scale, fallback_jump=40.0)
            )

            inner_radius = smooth_radius(inner_radius, max_jump=used_max_jump)
            outer_radius = smooth_radius(outer_radius, max_jump=used_max_jump)
            thickness = compute_thickness(inner_radius, outer_radius)
            if args.despike:
                thickness = despike_thickness(
                    thickness,
                    window=args.despike_window,
                    mad_thresh=args.despike_mad_thresh,
                )

            if args.thickness_median_kernel > 0:
                kernel = int(args.thickness_median_kernel)
                if kernel % 2 == 0:
                    kernel += 1
                if kernel >= 3:
                    thickness = medfilt(thickness, kernel_size=kernel)

            output_prefix = args.output_dir / image_path.stem
            outputs = visualize_results(
                image_bgr,
                edges,
                center,
                angles_deg,
                inner_radius,
                outer_radius,
                thickness,
                output_prefix,
                show=args.show,
                radial_line_step=args.radial_line_step,
                pixel_to_mm=args.pixel_to_mm,
            )

            _print_summary(
                image_path,
                center,
                method,
                thresholds,
                outputs,
                thickness,
                used_max_jump=used_max_jump,
                pixel_to_mm=args.pixel_to_mm,
            )
            processed_count += 1
        except Exception as exc:
            print(f"\nFailed to process {image_path.name}: {exc}")

    print(f"\nCompleted. Successfully processed {processed_count}/{len(image_paths)} image(s).")


if __name__ == "__main__":
    main()
