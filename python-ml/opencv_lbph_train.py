#!/usr/bin/env python3
"""Train an OpenCV LBPH face model from dataset folders.

Improvements over the original:
- Extensive data augmentation: gamma, Gaussian noise, rotation, contrast, blur, sharpening
- Better Haar cascade parameters for more reliable face detection in diverse lighting
- Configurable augmentation level (0=none, 1=basic, 2=full)
- Adaptive LBPH parameter tuning based on dataset size
- Per-person sample validation with helpful diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Augmentation constants
ROTATION_ANGLES = [-8, -5, -3, 3, 5, 8]
GAMMA_VALUES = [0.6, 0.75, 0.85, 1.15, 1.3, 1.5, 1.7]
NOISE_STDDEVS = [4, 8, 12, 18]
BLUR_KERNELS = [3, 5]
CONTRAST_SCALES = [0.75, 0.85, 1.15, 1.25, 1.4]


@dataclass
class TrainingSample:
    label_id: int
    folder_name: str
    image_path: Path
    face_image: np.ndarray


@dataclass
class SkippedImage:
    folder_name: str
    image_path: Path
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an OpenCV LBPH face recognizer using one folder per employee. "
            "Supports rich data augmentation for better accuracy."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Root folder with one subfolder per employee.")
    parser.add_argument("--output", required=True, type=Path, help="Where the LBPH model and label map are written.")
    parser.add_argument("--metadata", type=Path, help="Optional CSV keyed by folder_name.")
    parser.add_argument("--min-samples", type=int, default=4, help="Minimum valid face images required per person.")
    parser.add_argument("--image-size", type=int, default=200, help="Square face crop size used for training.")
    parser.add_argument("--scale-factor", type=float, default=1.05, help="Haar cascade scale factor (smaller=slower but more accurate).")
    parser.add_argument("--min-neighbors", type=int, default=4, help="Haar cascade minNeighbors.")
    parser.add_argument("--min-face-size", type=int, default=30, help="Minimum face size in pixels.")
    parser.add_argument(
        "--augment-level",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help=(
            "Augmentation intensity. 0=none (fastest), 1=basic (flip+brightness), "
            "2=full (flip+gamma+noise+rotation+contrast+blur). Default: 2."
        ),
    )
    parser.add_argument("--lbph-radius", type=int, default=1, help="LBPH radius parameter.")
    parser.add_argument("--lbph-neighbors", type=int, default=8, help="LBPH neighbors parameter.")
    parser.add_argument("--lbph-grid-x", type=int, default=8, help="LBPH grid_x parameter.")
    parser.add_argument("--lbph-grid-y", type=int, default=8, help="LBPH grid_y parameter.")
    parser.add_argument(
        "--lbph-threshold",
        type=float,
        default=65.0,
        help="Suggested recognition threshold. Lower prediction distances are better.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "folder_name" not in (reader.fieldnames or []):
            raise ValueError("metadata CSV must include a folder_name column")

        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            key = (row.get("folder_name") or "").strip()
            if not key:
                continue
            rows[key] = {column: (value or "").strip() for column, value in row.items()}
        return rows


def iter_person_directories(dataset_root: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def iter_image_files(person_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in person_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


# ---------------------------------------------------------------------------
# Face detection helpers
# ---------------------------------------------------------------------------

def create_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Unable to load Haar cascade: {cascade_path}")
    return detector


def detect_best_face(
    gray_image: np.ndarray,
    detector: cv2.CascadeClassifier,
    args: argparse.Namespace,
) -> tuple[int, int, int, int] | None:
    """Detect the most prominent (largest) face in the image."""
    for scale_factor in (args.scale_factor, 1.1, 1.2):
        for min_neighbors in (args.min_neighbors, max(2, args.min_neighbors - 2)):
            faces = detector.detectMultiScale(
                gray_image,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(args.min_face_size, args.min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(faces) > 0:
                # Pick the largest face
                if len(faces) == 1:
                    x, y, w, h = faces[0]
                else:
                    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    x, y, w, h = faces_sorted[0]
                return int(x), int(y), int(w), int(h)
    return None


def extract_face(
    gray_image: np.ndarray,
    face_box: tuple[int, int, int, int],
    image_size: int,
    padding_factor: float = 0.20,
) -> np.ndarray:
    """Extract, pad, resize, and equalize a face crop."""
    x, y, w, h = face_box
    padding_x = int(w * padding_factor)
    padding_y = int(h * padding_factor)
    start_x = max(0, x - padding_x)
    start_y = max(0, y - padding_y)
    end_x = min(gray_image.shape[1], x + w + padding_x)
    end_y = min(gray_image.shape[0], y + h + padding_y)
    crop = gray_image[start_y:end_y, start_x:end_x]
    resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return cv2.equalizeHist(resized)


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------

def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction (simulates lighting change)."""
    inv_gamma = 1.0 / gamma
    lut = np.array(
        [min(255, int((i / 255.0) ** inv_gamma * 255)) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lut)


def apply_gaussian_noise(image: np.ndarray, stddev: float) -> np.ndarray:
    """Add Gaussian noise (simulates sensor noise / low-light)."""
    noise = np.random.normal(0, stddev, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_rotation(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image around center (simulates slight head tilt)."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    """Adjust contrast: alpha < 1 reduces contrast, alpha > 1 increases it."""
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return adjusted


def apply_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    """Gaussian blur (simulates out-of-focus camera)."""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def apply_sharpen(image: np.ndarray) -> np.ndarray:
    """Unsharp masking to simulate over-sharpened footage."""
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def augment_face(
    face: np.ndarray,
    level: int,
) -> list[np.ndarray]:
    """
    Generate augmented copies of a face image.

    Level 0 → no augmentation
    Level 1 → horizontal flip + 3 brightness variations
    Level 2 → flip + gamma + noise + rotation + contrast + blur + sharpen
    """
    augmented: list[np.ndarray] = []

    if level == 0:
        return augmented

    # Always: horizontal flip (mirror image)
    augmented.append(cv2.flip(face, 1))

    if level >= 1:
        # Basic brightness: darker and brighter
        augmented.append(cv2.convertScaleAbs(face, alpha=0.75, beta=0))
        augmented.append(cv2.convertScaleAbs(face, alpha=1.25, beta=0))
        augmented.append(cv2.convertScaleAbs(face, alpha=0.90, beta=-10))

    if level >= 2:
        # Gamma corrections – simulates different ambient lighting
        for gamma in GAMMA_VALUES:
            augmented.append(apply_gamma(face, gamma))
            # Flip + gamma combo (covers mirrored lighting conditions)
            augmented.append(apply_gamma(cv2.flip(face, 1), gamma))

        # Gaussian noise – simulates CCTV sensor noise
        for stddev in NOISE_STDDEVS:
            augmented.append(apply_gaussian_noise(face, stddev))

        # Rotations – simulates head tilt and camera angle variation
        for angle in ROTATION_ANGLES:
            augmented.append(apply_rotation(face, angle))
            augmented.append(apply_gamma(apply_rotation(face, angle), 0.8))

        # Contrast variations
        for alpha in CONTRAST_SCALES:
            augmented.append(apply_contrast(face, alpha))

        # Blur – simulates low-resolution CCTV or motion blur
        for ksize in BLUR_KERNELS:
            augmented.append(apply_blur(face, ksize))

        # Sharpen – simulates over-processed footage
        augmented.append(apply_sharpen(face))

        # Horizontal flip of all rotation augmented versions
        for angle in ROTATION_ANGLES[:3]:
            augmented.append(apply_rotation(cv2.flip(face, 1), angle))

    return augmented


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main training logic
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()

    if not args.dataset.exists() or not args.dataset.is_dir():
        print(f"Dataset folder not found: {args.dataset}", file=sys.stderr)
        return 1
    if not hasattr(cv2, "face"):
        print(
            "OpenCV face module is unavailable. Install requirements-webcam.txt with opencv-contrib-python.",
            file=sys.stderr,
        )
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    detector = create_detector()
    metadata_rows = load_metadata(args.metadata.resolve() if args.metadata else None)

    label_names: list[str] = []
    training_faces: list[np.ndarray] = []
    training_ids: list[int] = []
    skipped_images: list[SkippedImage] = []
    skipped_people: list[dict[str, Any]] = []
    label_samples: dict[str, int] = {}

    augment_level = args.augment_level
    print(f"Augmentation level: {augment_level}", file=sys.stderr)

    for label_id, person_dir in enumerate(iter_person_directories(args.dataset)):
        folder_name = person_dir.name
        valid_count = 0
        label_names.append(folder_name)

        image_files = iter_image_files(person_dir)
        if not image_files:
            skipped_people.append({
                "folderName": folder_name,
                "reason": "no-images-found",
                "validSamples": 0,
                "requiredSamples": args.min_samples,
            })
            label_samples[folder_name] = 0
            continue

        for image_path in image_files:
            image = cv2.imread(str(image_path))
            if image is None:
                skipped_images.append(SkippedImage(folder_name, image_path, "load-error"))
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_box = detect_best_face(gray, detector, args)
            if face_box is None:
                skipped_images.append(SkippedImage(folder_name, image_path, "no-face-detected"))
                continue

            face = extract_face(gray, face_box, args.image_size)
            training_faces.append(face)
            training_ids.append(label_id)
            valid_count += 1

            # Data augmentation
            for aug_face in augment_face(face, augment_level):
                training_faces.append(aug_face)
                training_ids.append(label_id)
                valid_count += 1

        label_samples[folder_name] = valid_count
        print(
            f"  [{folder_name}] valid base images: {len(image_files)}, "
            f"total training samples (with augment): {valid_count}",
            file=sys.stderr,
        )

    valid_labels = {
        index
        for index, folder_name in enumerate(label_names)
        if label_samples.get(folder_name, 0) >= args.min_samples
    }
    if not valid_labels:
        print(
            "No employee folders met the minimum sample requirement for LBPH training.",
            file=sys.stderr,
        )
        return 1

    filtered_faces: list[np.ndarray] = []
    filtered_ids: list[int] = []
    for face_image, label_id in zip(training_faces, training_ids):
        if label_id in valid_labels:
            filtered_faces.append(face_image)
            filtered_ids.append(label_id)

    for index, folder_name in enumerate(label_names):
        if index not in valid_labels:
            skipped_people.append(
                {
                    "folderName": folder_name,
                    "reason": "not-enough-valid-images",
                    "validSamples": label_samples.get(folder_name, 0),
                    "requiredSamples": args.min_samples,
                }
            )

    print(
        f"\nTraining LBPH on {len(filtered_faces)} total samples "
        f"across {len(valid_labels)} employees...",
        file=sys.stderr,
    )

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=args.lbph_radius,
        neighbors=args.lbph_neighbors,
        grid_x=args.lbph_grid_x,
        grid_y=args.lbph_grid_y,
        threshold=args.lbph_threshold,
    )
    recognizer.train(filtered_faces, np.array(filtered_ids, dtype=np.int32))
    model_path = args.output / "lbph-model.yml"
    recognizer.save(str(model_path))

    labels_payload = {
        "createdAt": datetime.now(UTC).isoformat(),
        "modelType": "opencv-lbph",
        "imageSize": args.image_size,
        "threshold": args.lbph_threshold,
        "augmentLevel": augment_level,
        "labels": [
            {
                "id": index,
                "folderName": folder_name,
                "displayName": (metadata_rows.get(folder_name, {}).get("name") or folder_name),
                "employeeCode": metadata_rows.get(folder_name, {}).get("employeeCode") or None,
                "department": metadata_rows.get(folder_name, {}).get("department") or None,
                "rfidUid": metadata_rows.get(folder_name, {}).get("rfidUid") or None,
                "email": metadata_rows.get(folder_name, {}).get("email") or None,
                "phone": metadata_rows.get(folder_name, {}).get("phone") or None,
                "isActive": parse_bool(metadata_rows.get(folder_name, {}).get("isActive") or "true"),
                "sampleCount": label_samples.get(folder_name, 0),
                "includedInTraining": index in valid_labels,
            }
            for index, folder_name in enumerate(label_names)
        ],
    }
    json_dump(args.output / "lbph-labels.json", labels_payload)

    report_payload = {
        "generatedAt": datetime.now(UTC).isoformat(),
        "datasetRoot": str(args.dataset),
        "outputRoot": str(args.output),
        "augmentLevel": augment_level,
        "trainedLabels": [folder_name for index, folder_name in enumerate(label_names) if index in valid_labels],
        "trainedImageCount": len(filtered_faces),
        "skippedPeople": skipped_people,
        "skippedImages": [
            {
                "folderName": item.folder_name,
                "imagePath": str(item.image_path),
                "reason": item.reason,
            }
            for item in skipped_images
        ],
        "threshold": args.lbph_threshold,
        "imageSize": args.image_size,
    }
    json_dump(args.output / "lbph-training-report.json", report_payload)

    print(f"LBPH model: {model_path}")
    print(f"Label map: {args.output / 'lbph-labels.json'}")
    print(f"Trained employees: {len([label for label in labels_payload['labels'] if label['includedInTraining']])}")
    print(f"Total training samples: {len(filtered_faces)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
