"""Segmentation helpers exported from notebooks.

Provides:
- list_images, load_sample_image, show_images
- clean_mask, apply_grabcut, segment_leaf, mask_stats
- a small CLI to run batch segmentation and save masks/summaries

These routines mirror the logic used in `notebooks/segmentation.ipynb` so
the notebook can remain an interactive demo while this module provides
reusable, testable functionality and a CLI entrypoint for batch runs.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


DEFAULT_LOWER_GREEN = np.array([20, 30, 30], dtype=np.uint8)
DEFAULT_UPPER_GREEN = np.array([95, 255, 255], dtype=np.uint8)


def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS])


def load_sample_image(image_paths: List[Path], index: int = 0, resize_to: Tuple[int, int] = (512, 512)) -> Tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not image_paths:
        raise RuntimeError(f"No images found: {image_paths}")
    index = max(0, min(index, len(image_paths) - 1))
    image_path = image_paths[index]
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    image_bgr = cv2.resize(image_bgr, resize_to, interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return image_path, image_bgr, image_rgb, image_gray, image_hsv


def show_images(titles: List[str], images: List[np.ndarray], cmap_list: Optional[List[Optional[str]]] = None, figsize: Tuple[int, int] = (16, 8)) -> None:
    if cmap_list is None:
        cmap_list = [None] * len(images)
    cols = len(images)
    plt.figure(figsize=figsize)
    for i, (title, img, cmap) in enumerate(zip(titles, images, cmap_list), start=1):
        plt.subplot(1, cols, i)
        if cmap is None:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def clean_mask(mask: Optional[np.ndarray], kernel_size: int = 5, min_area_ratio: float = 0.01) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = int(np.argmax(areas)) + 1
    final = np.zeros_like(m)
    final[labels == max_idx] = 255
    contours, _ = cv2.findContours(final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(final)
    cv2.drawContours(mask_filled, contours, -1, 255, thickness=cv2.FILLED)
    img_area = mask.size
    if areas[max_idx - 1] < min_area_ratio * img_area:
        return m
    return mask_filled


def apply_grabcut(image_rgb: np.ndarray, init_mask: np.ndarray, iter_count: int = 5) -> np.ndarray:
    if init_mask is None:
        return None
    if init_mask.dtype != np.uint8:
        init_mask = (init_mask > 0).astype(np.uint8) * 255
    gc_mask = np.where(init_mask == 255, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype("uint8")
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image_rgb, gc_mask, None, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_MASK)
        res = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
        return res
    except Exception:
        return init_mask


def segment_leaf(image_rgb: np.ndarray, image_gray: np.ndarray, image_hsv: np.ndarray, *, n_clusters: int = 3, kernel_size: int = 5, lower_green: Optional[np.ndarray] = None, upper_green: Optional[np.ndarray] = None, grabcut_iter: int = 5, seed: int = 100) -> Dict:
    if lower_green is None:
        lower_green = DEFAULT_LOWER_GREEN
    if upper_green is None:
        upper_green = DEFAULT_UPPER_GREEN

    # Otsu thresholding
    _, mask_otsu = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(mask_otsu) > 127:
        mask_otsu = cv2.bitwise_not(mask_otsu)

    # HSV thresholding
    mask_hsv = cv2.inRange(image_hsv, lower_green, upper_green)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)

    # KMeans in LAB space
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape((-1, 3)).astype(np.float32)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers_lab = kmeans.cluster_centers_.astype(np.uint8)
    centers_lab_img = centers_lab.reshape(1, -1, 3)
    centers_rgb = cv2.cvtColor(centers_lab_img, cv2.COLOR_LAB2RGB).reshape(-1, 3)
    green_score = centers_rgb[:, 1] - (centers_rgb[:, 0] + centers_rgb[:, 2]) / 2
    leaf_cluster = int(np.argmax(green_score))
    mask_kmeans = (labels.reshape(image_rgb.shape[:2]) == leaf_cluster).astype(np.uint8) * 255
    mask_kmeans = cv2.morphologyEx(mask_kmeans, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # combine and clean
    combined = cv2.bitwise_or(mask_hsv, mask_otsu)
    combined = cv2.bitwise_or(combined, mask_kmeans)
    cleaned = clean_mask(combined, kernel_size=kernel_size)

    img_area = cleaned.size
    if cleaned.sum() < 0.01 * img_area * 255:
        grab = apply_grabcut(image_rgb, cleaned, iter_count=grabcut_iter)
        final = clean_mask(grab, kernel_size=kernel_size)
    else:
        final = cleaned

    leaf_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=final)
    return {
        "mask_otsu": mask_otsu,
        "mask_hsv": mask_hsv,
        "mask_kmeans": mask_kmeans,
        "combined": combined,
        "refined": final,
        "leaf_refined": leaf_rgb,
        "kmeans_centers_rgb": centers_rgb,
        "leaf_cluster": leaf_cluster,
    }


def mask_stats(mask: Optional[np.ndarray]) -> Dict:
    if mask is None:
        return {"area": 0, "area_fraction": 0.0, "n_contours": 0}
    bin_mask = (mask > 0).astype(np.uint8)
    area = int(bin_mask.sum())
    area_frac = area / mask.size
    contours, _ = cv2.findContours(bin_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return {"area": area, "area_fraction": float(area_frac), "n_contours": int(len(contours))}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch segmentation CLI: segment leaves and save masks/summaries.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input dataset root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output results root")
    parser.add_argument("--resize", type=int, nargs=2, default=[512, 512], metavar=("W", "H"), help="Resize images to this size")
    parser.add_argument("--samples-per-folder", type=int, default=1, help="Number of samples per folder to process")
    parser.add_argument("--n-clusters", type=int, default=3, help="KMeans clusters")
    parser.add_argument("--kernel-size", type=int, default=5, help="Morphology kernel size")
    parser.add_argument("--grabcut-iter", type=int, default=5, help="GrabCut iterations")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--lower-green", type=int, nargs=3, default=[20, 30, 30], help="Lower HSV bound for green (H S V)")
    parser.add_argument("--upper-green", type=int, nargs=3, default=[95, 255, 255], help="Upper HSV bound for green (H S V)")
    parser.add_argument("--single-image", type=Path, default=None, help="Process a single image path instead of batch")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    resize_to = (int(args.resize[0]), int(args.resize[1]))
    output_dir.mkdir(parents=True, exist_ok=True)

    lower_green = np.array(list(map(int, args.lower_green)), dtype=np.uint8)
    upper_green = np.array(list(map(int, args.upper_green)), dtype=np.uint8)

    if args.single_image:
        image_paths = [args.single_image]
    else:
        image_paths = list_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {input_dir}")

    # Batch by folder
    folders = sorted({p.parent for p in image_paths})
    batch_summary = []
    for folder in folders:
        imgs = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
        if not imgs:
            continue
        samples = imgs[: args.samples_per_folder]
        for sample_idx, img_path in enumerate(samples, start=1):
            try:
                image_bgr = cv2.imread(str(img_path))
                if image_bgr is None:
                    print(f"Cannot read {img_path}, skipping")
                    continue
                image_bgr = cv2.resize(image_bgr, resize_to, interpolation=cv2.INTER_AREA)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

                sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel_mag = cv2.magnitude(sobel_x, sobel_y)
                sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                canny_edges = cv2.Canny(image_gray, 70, 150)

                seg_results = segment_leaf(image_rgb, image_gray, image_hsv, n_clusters=args.n_clusters, kernel_size=args.kernel_size, lower_green=lower_green, upper_green=upper_green, grabcut_iter=args.grabcut_iter, seed=args.seed)

                mask_otsu = seg_results["mask_otsu"]
                mask_hsv = seg_results["mask_hsv"]
                mask_kmeans = seg_results["mask_kmeans"]
                mask_combined = seg_results["combined"]
                mask_refined = seg_results["refined"]
                leaf_otsu = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_otsu)
                leaf_hsv = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_hsv)
                leaf_kmeans = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_kmeans)
                leaf_refined = seg_results["leaf_refined"]

                base = f"{folder.name}_{Path(img_path).stem}_{sample_idx}"
                cv2.imwrite(str(output_dir / f"{base}_mask_otsu.png"), mask_otsu)
                cv2.imwrite(str(output_dir / f"{base}_mask_hsv.png"), mask_hsv)
                cv2.imwrite(str(output_dir / f"{base}_mask_kmeans.png"), mask_kmeans)
                cv2.imwrite(str(output_dir / f"{base}_mask_combined.png"), mask_combined)
                cv2.imwrite(str(output_dir / f"{base}_mask_refined.png"), mask_refined)
                cv2.imwrite(str(output_dir / f"{base}_sobel.png"), sobel_mag)
                cv2.imwrite(str(output_dir / f"{base}_canny.png"), canny_edges)
                cv2.imwrite(str(output_dir / f"{base}_leaf_otsu.png"), cv2.cvtColor(leaf_otsu, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(output_dir / f"{base}_leaf_hsv.png"), cv2.cvtColor(leaf_hsv, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(output_dir / f"{base}_leaf_kmeans.png"), cv2.cvtColor(leaf_kmeans, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(output_dir / f"{base}_leaf_refined.png"), cv2.cvtColor(leaf_refined, cv2.COLOR_RGB2BGR))

                area_refined = int((mask_refined > 0).astype(np.uint8).sum())
                area_frac = float(area_refined) / mask_refined.size if mask_refined is not None else 0.0
                batch_summary.append({
                    "folder": str(folder),
                    "sample": str(img_path),
                    "leaf_cluster": int(seg_results.get("leaf_cluster", -1)),
                    "refined_area_fraction": area_frac,
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    summary_path = output_dir / "one_per_folder_summary.json"
    summary_path.write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    print(f"Batch processing complete. Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
