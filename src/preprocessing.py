import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.features import extract_features


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess image dataset: resize, color conversion, denoising, histogram export."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Input dataset root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset root.")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("WIDTH", "HEIGHT"),
        help="Output image size (width height). Default: 224 224.",
    )
    parser.add_argument(
        "--color-mode",
        choices=["rgb", "hsv", "gray"],
        default="rgb",
        help="Output color mode. Default: rgb.",
    )
    parser.add_argument(
        "--denoise",
        choices=["none", "gaussian", "median", "bilateral"],
        default="none",
        help="Denoising filter. Default: none.",
    )
    parser.add_argument(
        "--hist-samples",
        type=int,
        default=8,
        help="Number of sample images for histogram analysis. Default: 8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for histogram sample selection. Default: 42.",
    )
    parser.add_argument(
        "--extract-features",
        action="store_true",
        help="Compute and save features (color, texture, shape) for each image.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="Directory to store extracted feature files. Default: <output-dir>/features",
    )
    parser.add_argument(
        "--auto-mask",
        action="store_true",
        help="Automatically generate a binary mask (Otsu) to compute shape features.",
    )
    return parser.parse_args()


def list_image_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]


def apply_denoise(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return image
    if mode == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    if mode == "median":
        return cv2.medianBlur(image, 5)
    if mode == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    raise ValueError(f"Unsupported denoise mode: {mode}")


def convert_color(image_bgr: np.ndarray, color_mode: str) -> np.ndarray:
    if color_mode == "rgb":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if color_mode == "hsv":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if color_mode == "gray":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported color mode: {color_mode}")


def generate_auto_mask_from_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """Quick auto-mask using grayscale Otsu threshold and morphological closing.

    Returns a boolean mask (HxW) where True indicates foreground.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ensure foreground is white; if background is white, invert
    # heuristic: check mean intensity of mask
    if th.mean() > 127:
        th = cv2.bitwise_not(th)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed.astype(bool)


def save_output_image(image: np.ndarray, output_path: Path, color_mode: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if color_mode == "gray":
        cv2.imwrite(str(output_path), image)
        return

    if color_mode == "rgb":
        image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), image_to_save)
        return

    # OpenCV stores HSV arrays directly; this keeps HSV data for downstream analysis.
    cv2.imwrite(str(output_path), image)


def plot_histogram(image: np.ndarray, image_name: str, color_mode: str, save_path: Path) -> None:
    plt.figure(figsize=(9, 4))

    if color_mode == "gray":
        plt.hist(image.ravel(), bins=256, range=(0, 256), color="black")
        plt.title(f"Histogram (grayscale) - {image_name}")
        plt.xlabel("Intensity")
        plt.ylabel("Pixel count")
    else:
        channel_names = ["C1", "C2", "C3"]
        channel_colors = ["r", "g", "b"]
        for idx in range(3):
            hist = cv2.calcHist([image], [idx], None, [256], [0, 256])
            plt.plot(hist, color=channel_colors[idx], label=channel_names[idx])
        plt.title(f"Histogram ({color_mode.upper()}) - {image_name}")
        plt.xlabel("Value")
        plt.ylabel("Pixel count")
        plt.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    width, height = args.size

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = list_image_files(input_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {input_dir}")

    np.random.seed(args.seed)
    histogram_samples = set(np.random.choice(len(image_paths), size=min(args.hist_samples, len(image_paths)), replace=False))

    processed_count = 0
    features_saved = 0
    features_out_dir: Path | None = None
    if args.extract_features:
        features_out_dir = args.features_dir.resolve() if args.features_dir is not None else output_dir / "features"
        features_out_dir.mkdir(parents=True, exist_ok=True)
    failed_files: list[str] = []

    for index, image_path in enumerate(image_paths):
        relative_path = image_path.relative_to(input_dir)
        output_path = output_dir / relative_path

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            failed_files.append(str(relative_path))
            continue

        image_bgr = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
        image_bgr = apply_denoise(image_bgr, args.denoise)
        image_processed = convert_color(image_bgr, args.color_mode)

        save_output_image(image_processed, output_path, args.color_mode)
        processed_count += 1

        if args.extract_features:
            # ensure we compute features on RGB image regardless of output color mode
            rgb_img = convert_color(image_bgr, "rgb")
            mask = None
            if args.auto_mask:
                mask = generate_auto_mask_from_bgr(image_bgr)
            try:
                vector, details = extract_features(rgb_img, mask=mask)
                feat_path = (features_out_dir / relative_path).with_suffix(".json")
                feat_path.parent.mkdir(parents=True, exist_ok=True)
                feat_json = {
                    "image": str(relative_path),
                    "vector": vector.tolist(),
                    "details": {
                        "rgb_hist": details["rgb_hist"].tolist(),
                        "hsv_hist": details["hsv_hist"].tolist(),
                        "glcm": details["glcm"],
                        "shape": details["shape"],
                    },
                }
                feat_path.write_text(json.dumps(feat_json, indent=2), encoding="utf-8")
                features_saved += 1
            except Exception as e:
                failed_files.append(str(relative_path))
                print(f"Feature extraction failed for {relative_path}: {e}")

        if index in histogram_samples:
            hist_name = output_path.stem + "_hist.png"
            hist_path = output_dir / "histograms" / hist_name
            plot_histogram(image_processed, output_path.name, args.color_mode, hist_path)

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "target_size": [width, height],
        "color_mode": args.color_mode,
        "denoise": args.denoise,
        "total_images_found": len(image_paths),
        "images_processed": processed_count,
        "images_failed": len(failed_files),
        "failed_files": failed_files,
    }
    if args.extract_features:
        report["features_saved"] = features_saved
        report["features_dir"] = str(features_out_dir)

    report_path = output_dir / "preprocessing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Preprocessing complete.")
    print(f"Processed: {processed_count}/{len(image_paths)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()