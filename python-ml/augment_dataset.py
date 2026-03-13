#!/usr/bin/env python3
"""
Augment employee dataset images for better LBPH training.

This script generates a rich set of augmented training images from raw
employee face photos. It is designed to run before opencv_lbph_train.py
to maximise dataset size and diversity.

Changes from the original:
- Uses CLAHE equalization for better local contrast handling
- Wider augmentation: gamma, noise, rotation, blur, contrast, sharpening
- Proper face detection before augmentation (only augments detected faces)
- CSV-driven employee folder lookup
- 50 augmented copies per base image (up from 30)
"""

from __future__ import annotations

import csv
import os
import sys
import urllib.request
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore

DATASET_DIR = Path("dataset")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUG_PER_IMAGE = 50  # augmented copies per base image


def get_employee_folders() -> list[str]:
    folders: list[str] = []
    if Path("metadata.generated.csv").exists():
        with open("metadata.generated.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                folder = (row.get("folder_name") or "").strip()
                if folder:
                    folders.append(folder)
    if not folders:
        folders = ["emp-1"]
    return folders


def download_face(url: str, save_path: Path) -> bool:
    print(f"Downloading {url} → {save_path}", file=sys.stderr)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp, open(save_path, "wb") as out:
            out.write(resp.read())
        return True
    except Exception as exc:
        print(f"  Download failed: {exc}", file=sys.stderr)
        return False


def create_clahe() -> cv2.CLAHE:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / max(gamma, 1e-6)
    lut = np.array(
        [min(255, int((i / 255.0) ** inv_gamma * 255)) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lut)


def apply_gaussian_noise(image: np.ndarray, stddev: float) -> np.ndarray:
    noise = np.random.normal(0, stddev, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_rotation(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_sharpen(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    return np.clip(cv2.addWeighted(image, 1.5, blurred, -0.5, 0), 0, 255).astype(np.uint8)


def detect_and_crop_face(
    img: np.ndarray,
    detector: cv2.CascadeClassifier,
    image_size: int = 200,
    padding_factor: float = 0.20,
) -> np.ndarray | None:
    """Detect the largest face, crop with padding, resize and equalize."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    for sf, mn in [(1.05, 4), (1.1, 3), (1.2, 2)]:
        faces = detector.detectMultiScale(gray, scaleFactor=sf, minNeighbors=mn, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            px = int(w * padding_factor)
            py = int(h * padding_factor)
            x1 = max(0, x - px)
            y1 = max(0, y - py)
            x2 = min(gray.shape[1], x + w + px)
            y2 = min(gray.shape[0], y + h + py)
            crop = gray[y1:y2, x1:x2]
            resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            clahe = create_clahe()
            return clahe.apply(resized)
    return None


def augment_face_image(face: np.ndarray, count: int) -> list[np.ndarray]:
    """Generate `count` augmented copies of a grayscale face image."""
    augmented: list[np.ndarray] = []

    # Augmentation recipes to draw from
    def recipe_flip():       return cv2.flip(face, 1)
    def recipe_rot(a):       return apply_rotation(face, a)
    def recipe_rot_flip(a):  return apply_rotation(cv2.flip(face, 1), a)
    def recipe_gamma(g):     return apply_gamma(face, g)
    def recipe_noise(s):     return apply_gaussian_noise(face, s)
    def recipe_blur(k):      return cv2.GaussianBlur(face, (k, k), 0)
    def recipe_contrast(a):  return cv2.convertScaleAbs(face, alpha=a, beta=0)
    def recipe_sharpen():    return apply_sharpen(face)
    def recipe_gamma_flip(g):return apply_gamma(cv2.flip(face, 1), g)
    def recipe_combined(g, a): return apply_gamma(cv2.convertScaleAbs(face, alpha=a, beta=0), g)

    gammas = [0.55, 0.65, 0.75, 0.85, 1.15, 1.30, 1.50, 1.70, 1.90]
    angles = [-10, -7, -5, -3, 3, 5, 7, 10]
    noises = [4, 8, 12, 18, 24]
    blurs = [3, 5]
    contrasts = [0.70, 0.80, 0.90, 1.10, 1.20, 1.35, 1.50]

    recipes = (
        [recipe_flip, recipe_sharpen]
        + [lambda a=a: recipe_rot(a) for a in angles]
        + [lambda a=a: recipe_rot_flip(a) for a in angles]
        + [lambda g=g: recipe_gamma(g) for g in gammas]
        + [lambda g=g: recipe_gamma_flip(g) for g in gammas]
        + [lambda s=s: recipe_noise(s) for s in noises]
        + [lambda k=k: recipe_blur(k) for k in blurs]
        + [lambda a=a: recipe_contrast(a) for a in contrasts]
        + [lambda g=g, a=a: recipe_combined(g, a) for g in gammas[:4] for a in contrasts[:3]]
    )

    # Shuffle so we get a random diverse mix when count < total recipes
    order = list(range(len(recipes)))
    np.random.shuffle(order)
    for i in order[:count]:
        try:
            aug = recipes[i]()
            if aug is not None:
                augmented.append(aug)
        except Exception:
            pass

    return augmented


def augment_image_file(img_path: Path, output_dir: Path, detector: cv2.CascadeClassifier, count: int = AUG_PER_IMAGE) -> int:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Could not read {img_path}", file=sys.stderr)
        return 0

    face = detect_and_crop_face(img, detector)
    if face is None:
        # Fall back to the full image if no face detected (some profile shots)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = create_clahe()
        face = clahe.apply(cv2.resize(gray, (200, 200)))

    augmented = augment_face_image(face, count)

    stem = img_path.stem
    written = 0
    for idx, aug in enumerate(augmented):
        out_path = output_dir / f"{stem}_aug_{idx:03d}.jpg"
        if cv2.imwrite(str(out_path), aug):
            written += 1
    return written


def main() -> int:
    DATASET_DIR.mkdir(exist_ok=True)
    folders = get_employee_folders()

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        print(f"Unable to load Haar cascade: {cascade_path}", file=sys.stderr)
        return 1

    for i, emp_folder in enumerate(folders):
        emp_dir = DATASET_DIR / emp_folder
        emp_dir.mkdir(exist_ok=True)

        base_img_path = emp_dir / "base.jpg"
        if not base_img_path.exists():
            url = f"https://randomuser.me/api/portraits/men/{i + 10}.jpg"
            if not download_face(url, base_img_path):
                print(f"  Skipping {emp_folder} – no base image.", file=sys.stderr)
                continue

        # Find all raw images (non-augmented)
        raw_images = [
            p for p in emp_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and "_aug_" not in p.stem
        ]

        if not raw_images:
            print(f"  No raw images found for {emp_folder}.", file=sys.stderr)
            continue

        total_written = 0
        for raw_img in raw_images:
            written = augment_image_file(raw_img, emp_dir, detector, count=AUG_PER_IMAGE)
            total_written += written

        print(f"[{emp_folder}] Generated {total_written} augmented images from {len(raw_images)} raw image(s).")

    print("\nAugmentation complete. Run opencv_lbph_train.py to train the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
