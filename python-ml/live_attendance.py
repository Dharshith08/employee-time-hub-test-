#!/usr/bin/env python3
"""Run live webcam or RTSP attendance using LBPH face profiles.

Improvements over the original (face_recognition-based) version:
- Uses OpenCV LBPH (via the trained model) instead of dlib/face_recognition
  so it matches the same model used by the web gate verification
- CLAHE equalization and multi-scale NMS face detection
- Automatic RTSP reconnection with exponential back-off
- Fixed --no-display logic (original had a misplaced break/continue)
- Configurable confidence thresholds
- Optional webhook/HTTP posting of attendance events
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import numpy as np  # type: ignore

# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------
LBPH_DISTANCE_THRESHOLD = 65.0   # lower = stricter
MIN_CONSECUTIVE_DETECTIONS = 2

EVENT_FIELDS = [
    "timestamp",
    "source",
    "employeeLabel",
    "displayName",
    "employeeCode",
    "department",
    "rfidUid",
    "distance",
    "confidence",
    "sampleCount",
    "faceIndex",
    "top",
    "right",
    "bottom",
    "left",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LBPHProfile:
    folder_name: str
    display_name: str
    sample_count: int | None
    employee_code: str | None
    department: str | None
    rfid_uid: str | None
    label_id: int


@dataclass
class DetectionResult:
    face_index: int
    box: tuple[int, int, int, int]   # top, right, bottom, left (like face_recognition)
    verified: bool
    label: LBPHProfile | None
    distance: float | None
    confidence: float


@dataclass
class SessionAttendance:
    first_seen_at: str
    last_seen_at: str
    times_marked: int
    display_name: str
    employee_code: str | None
    department: str | None
    rfid_uid: str | None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run live attendance from a webcam or RTSP CCTV stream using the trained LBPH model. "
            "Use --source 0 for the laptop webcam."
        )
    )
    parser.add_argument("--model", required=True, type=Path, help="Path to lbph-model.yml")
    parser.add_argument("--labels", required=True, type=Path, help="Path to lbph-labels.json")
    parser.add_argument("--source", default="0", help="Video source: 0 for webcam or an RTSP URL.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/live-attendance"),
        help="Where attendance logs, summaries, and optional snapshots will be written.",
    )
    parser.add_argument("--max-faces", type=int, default=12, help="Maximum faces to process per frame.")
    parser.add_argument(
        "--process-every-nth-frame",
        type=int,
        default=3,
        help="Process every Nth frame to keep webcam performance smooth.",
    )
    parser.add_argument(
        "--min-consecutive-detections",
        type=int,
        default=MIN_CONSECUTIVE_DETECTIONS,
        help="How many processed frames a person must appear in before attendance is marked.",
    )
    parser.add_argument(
        "--repeat-after-seconds",
        type=float,
        default=0.0,
        help="0 = mark once per session. Set >0 to allow repeats after a cooldown.",
    )
    parser.add_argument("--resize-width", type=int, default=960, help="Resize frames before detection. 0 to disable.")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--scale-factor", type=float, default=1.05, help="Haar cascade scale factor.")
    parser.add_argument("--min-neighbors", type=int, default=4, help="Haar cascade minNeighbors.")
    parser.add_argument("--min-face-size", type=int, default=55, help="Minimum face size in pixels.")
    parser.add_argument("--distance-threshold", type=float, default=LBPH_DISTANCE_THRESHOLD, help="LBPH distance threshold.")
    parser.add_argument("--save-snapshots", action="store_true", help="Save a JPEG whenever attendance is marked.")
    parser.add_argument("--no-display", action="store_true", help="Run headless (no OpenCV window).")
    parser.add_argument(
        "--webhook-url",
        type=str,
        default=None,
        help="Optional HTTP endpoint to POST attendance events as JSON.",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=3.0,
        help="Seconds to wait before reconnecting to an RTSP source on failure.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# LBPH helpers
# ---------------------------------------------------------------------------

def load_lbph_model(model_path: Path, labels_path: Path) -> tuple[Any, dict[int, LBPHProfile], float]:
    if not hasattr(cv2, "face"):
        raise RuntimeError("opencv-contrib-python is required for LBPH recognition.")

    labels_payload = json.loads(labels_path.resolve().read_text(encoding="utf-8"))
    threshold = float(labels_payload.get("threshold", LBPH_DISTANCE_THRESHOLD))

    profiles: dict[int, LBPHProfile] = {}
    for item in labels_payload.get("labels", []):
        if not item.get("includedInTraining", True):
            continue
        label_id = int(item["id"])
        profiles[label_id] = LBPHProfile(
            folder_name=item.get("folderName") or f"label-{label_id}",
            display_name=item.get("displayName") or item.get("folderName") or f"label-{label_id}",
            sample_count=item.get("sampleCount"),
            employee_code=item.get("employeeCode"),
            department=item.get("department"),
            rfid_uid=item.get("rfidUid"),
            label_id=label_id,
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_path.resolve()))
    return recognizer, profiles, threshold


def distance_to_confidence(distance: float, threshold: float) -> float:
    """Sigmoid-based distance → [0, 1] confidence."""
    k = 6.0 / max(threshold, 1.0)
    sigmoid = 1.0 / (1.0 + np.exp(k * (distance - 0.6 * threshold)))
    return float(np.clip(sigmoid, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Face detection helpers
# ---------------------------------------------------------------------------

def create_cascade() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Unable to load Haar cascade: {cascade_path}")
    return cascade


def detect_faces_nms(
    gray: np.ndarray,
    cascade: cv2.CascadeClassifier,
    scale_factor: float,
    min_neighbors: int,
    min_face_size: int,
) -> list[tuple[int, int, int, int]]:
    """Multi-scale detection + NMS, returns (x, y, w, h) sorted by area desc."""
    all_faces: list[tuple[int, int, int, int]] = []
    for sf in (scale_factor, min(scale_factor + 0.05, 1.3)):
        for mn in (min_neighbors, max(2, min_neighbors - 2)):
            faces = cascade.detectMultiScale(
                gray, scaleFactor=sf, minNeighbors=mn,
                minSize=(min_face_size, min_face_size), flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(faces) > 0:
                all_faces.extend((int(x), int(y), int(w), int(h)) for x, y, w, h in faces)

    if not all_faces:
        return []

    # NMS
    sorted_faces = sorted(all_faces, key=lambda f: f[2] * f[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    suppressed = [False] * len(sorted_faces)
    for i, fa in enumerate(sorted_faces):
        if suppressed[i]:
            continue
        kept.append(fa)
        for j in range(i + 1, len(sorted_faces)):
            if not suppressed[j] and _iou(fa, sorted_faces[j]) > 0.35:
                suppressed[j] = True

    return kept


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx); iy = max(ay, by)
    ir = min(ax + aw, bx + bw); ib = min(ay + ah, by + bh)
    iw = max(0, ir - ix); ih_v = max(0, ib - iy)
    intersection = iw * ih_v
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def prepare_face_crop(gray: np.ndarray, x: int, y: int, w: int, h: int, image_size: int = 200) -> np.ndarray:
    px = int(w * 0.20); py = int(h * 0.20)
    x1 = max(0, x - px); y1 = max(0, y - py)
    x2 = min(gray.shape[1], x + w + px); y2 = min(gray.shape[0], y + h + py)
    crop = gray[y1:y2, x1:x2]
    resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(resized)


# ---------------------------------------------------------------------------
# Frame recognition using LBPH
# ---------------------------------------------------------------------------

def recognize_frame_lbph(
    frame: np.ndarray,
    recognizer: Any,
    profiles: dict[int, LBPHProfile],
    args: argparse.Namespace,
    threshold: float,
) -> list[DetectionResult]:
    # Resize for speed
    if args.resize_width > 0:
        h, w = frame.shape[:2]
        if w > args.resize_width:
            scale = args.resize_width / float(w)
            frame_small = cv2.resize(frame, (args.resize_width, max(1, int(h * scale))))
            inv_scale = 1.0 / scale
        else:
            frame_small = frame
            inv_scale = 1.0
    else:
        frame_small = frame
        inv_scale = 1.0

    gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_small = detect_faces_nms(
        gray_small, _cascade, args.scale_factor, args.min_neighbors, args.min_face_size
    )
    faces_small = faces_small[: args.max_faces]

    detections: list[DetectionResult] = []
    for idx, (x, y, w, h) in enumerate(faces_small):
        # Scale back to full-resolution coords
        fx = int(round(x * inv_scale)); fy = int(round(y * inv_scale))
        fw = int(round(w * inv_scale)); fh = int(round(h * inv_scale))

        face_crop = prepare_face_crop(gray_full, fx, fy, fw, fh)
        label_id_raw, distance = recognizer.predict(face_crop)
        label_id = int(label_id_raw)
        distance = float(distance)

        profile = profiles.get(label_id)
        verified = profile is not None and distance <= threshold
        confidence = distance_to_confidence(distance, threshold) if profile else 0.0

        # Convert to top/right/bottom/left like face_recognition
        top = fy; right = fx + fw; bottom = fy + fh; left = fx
        detections.append(DetectionResult(
            face_index=idx,
            box=(top, right, bottom, left),
            verified=verified,
            label=profile if verified else None,
            distance=distance,
            confidence=confidence,
        ))

    return detections


# ---------------------------------------------------------------------------
# Capture / CSV / session helpers
# ---------------------------------------------------------------------------

def ensure_csv_header(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=EVENT_FIELDS).writeheader()


def append_event(path: Path, row: dict[str, Any]) -> None:
    ensure_csv_header(path)
    with path.open("a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=EVENT_FIELDS).writerow(row)


def should_mark(
    label: str,
    now_ts: float,
    logged_once: set[str],
    last_logged_at: dict[str, float],
    repeat_after: float,
) -> bool:
    if repeat_after <= 0:
        return label not in logged_once
    last = last_logged_at.get(label)
    return last is None or (now_ts - last) >= repeat_after


def post_webhook(url: str, payload: dict[str, Any]) -> None:
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3):
            pass
    except Exception as exc:
        print(f"[webhook] POST failed: {exc}", file=sys.stderr)


def open_capture(source_value: int | str, args: argparse.Namespace) -> cv2.VideoCapture:
    if isinstance(source_value, int):
        cap = cv2.VideoCapture(source_value, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(source_value)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        return cap
    # RTSP
    cap = cv2.VideoCapture(source_value)
    return cap


def draw_overlay(
    frame: np.ndarray,
    detections: list[DetectionResult],
    roster_size: int,
    attendance_count: int,
    source_label: str,
    status_message: str,
) -> np.ndarray:
    canvas = frame.copy()
    for det in detections:
        top, right, bottom, left = det.box
        if det.verified and det.label:
            color = (40, 200, 60)
            label_text = f"{det.label.display_name} {det.confidence:.0%}"
        else:
            color = (30, 30, 210)
            dist_str = f"d={det.distance:.1f}" if det.distance is not None else ""
            label_text = f"Unknown {dist_str}"
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
        cv2.rectangle(canvas, (left, max(0, top - 28)), (right, top), color, -1)
        cv2.putText(canvas, label_text[:42], (left + 6, max(18, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    for i, line in enumerate([
        f"Source: {source_label}", f"Roster: {roster_size}", f"Marked: {attendance_count}", status_message
    ]):
        cv2.putText(canvas, line, (16, 28 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
    return canvas


def write_summary(
    path: Path,
    source_label: str,
    roster_size: int,
    session_started_at: str,
    session_attendance: dict[str, SessionAttendance],
) -> None:
    path.write_text(json.dumps({
        "sessionStartedAt": session_started_at,
        "sessionEndedAt": datetime.now(UTC).isoformat(),
        "source": source_label,
        "rosterSize": roster_size,
        "attendanceMarked": len(session_attendance),
        "employees": [
            {
                "employeeLabel": label,
                "displayName": item.display_name,
                "employeeCode": item.employee_code,
                "department": item.department,
                "rfidUid": item.rfid_uid,
                "firstSeenAt": item.first_seen_at,
                "lastSeenAt": item.last_seen_at,
                "timesMarked": item.times_marked,
            }
            for label, item in sorted(session_attendance.items())
        ],
    }, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Global cascade (lazy-initialised)
# ---------------------------------------------------------------------------
_cascade: cv2.CascadeClassifier | None = None  # type: ignore[assignment]


def get_cascade() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is None:
        _cascade = create_cascade()
    return _cascade


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    global _cascade
    args: Any = parse_args()
    _cascade = create_cascade()

    recognizer, profiles, threshold = load_lbph_model(args.model, args.labels)
    if not profiles:
        print("No trained profiles loaded from labels JSON.", file=sys.stderr)
        return 1

    args.distance_threshold = threshold if args.distance_threshold == LBPH_DISTANCE_THRESHOLD else args.distance_threshold

    source_str = args.source
    source_value: int | str = int(source_str) if source_str.isdigit() else source_str
    source_label = f"webcam:{source_value}" if isinstance(source_value, int) else str(source_value)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_log_path = output_dir / "attendance-events.csv"
    summary_path = output_dir / "session-summary.json"
    snapshot_dir = output_dir / "snapshots"

    session_started_at = datetime.now(UTC).isoformat()
    frame_counter = 0
    consecutive_hits: dict[str, int] = {}
    logged_once: set[str] = set()
    last_logged_at: dict[str, float] = {}
    session_attendance: dict[str, SessionAttendance] = {}
    latest_detections: list[DetectionResult] = []
    status_message = "Show your face to the camera. Press Q to quit."
    reconnect_delay = float(args.reconnect_delay)
    is_rtsp = not isinstance(source_value, int)

    print(f"[live-attendance] Starting. Source: {source_label} | Threshold: {args.distance_threshold}", file=sys.stderr)

    capture = open_capture(source_value, args)
    if not capture.isOpened():
        print(f"Unable to open video source: {args.source}", file=sys.stderr)
        return 1

    try:
        while True:
            ok, frame = capture.read()

            # Handle read failure (e.g. RTSP disconnect)
            if not ok or frame is None:
                status_message = "Frame read failed."
                print(f"[live-attendance] {status_message}", file=sys.stderr)
                if is_rtsp:
                    capture.release()
                    print(f"[live-attendance] Reconnecting in {reconnect_delay:.0f}s...", file=sys.stderr)
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, 30.0)  # exponential back-off, max 30s
                    capture = open_capture(source_value, args)
                    continue
                else:
                    if args.no_display:
                        break
                    # Webcam: keep trying
                    time.sleep(0.1)
                    continue
            else:
                reconnect_delay = float(args.reconnect_delay)  # reset on success

            frame_counter += 1
            nth = max(1, int(args.process_every_nth_frame))
            if frame_counter % nth == 0:
                latest_detections = recognize_frame_lbph(frame, recognizer, profiles, args, args.distance_threshold)

                current_labels = {
                    det.label.folder_name
                    for det in latest_detections
                    if det.verified and det.label is not None
                }
                for lbl in list(consecutive_hits.keys()):
                    if lbl not in current_labels:
                        consecutive_hits[lbl] = 0

                now_ts = time.time()
                now_iso = datetime.now(UTC).isoformat()
                for det in latest_detections:
                    if not det.verified or det.label is None:
                        continue
                    lbl = det.label.folder_name
                    consecutive_hits[lbl] = consecutive_hits.get(lbl, 0) + 1
                    min_req = max(1, int(args.min_consecutive_detections))
                    if consecutive_hits[lbl] < min_req:
                        continue
                    if not should_mark(lbl, now_ts, logged_once, last_logged_at, args.repeat_after_seconds):
                        continue

                    top, right, bottom, left = det.box
                    event_row = {
                        "timestamp": now_iso,
                        "source": source_label,
                        "employeeLabel": lbl,
                        "displayName": det.label.display_name,
                        "employeeCode": det.label.employee_code or "",
                        "department": det.label.department or "",
                        "rfidUid": det.label.rfid_uid or "",
                        "distance": round(det.distance, 3) if det.distance is not None else "",
                        "confidence": round(det.confidence, 4),
                        "sampleCount": det.label.sample_count or "",
                        "faceIndex": det.face_index,
                        "top": top, "right": right, "bottom": bottom, "left": left,
                    }
                    append_event(event_log_path, event_row)

                    existing = session_attendance.get(lbl)
                    if existing is None:
                        session_attendance[lbl] = SessionAttendance(
                            first_seen_at=now_iso, last_seen_at=now_iso, times_marked=1,
                            display_name=det.label.display_name, employee_code=det.label.employee_code,
                            department=det.label.department, rfid_uid=det.label.rfid_uid,
                        )
                    else:
                        existing.last_seen_at = now_iso
                        existing.times_marked += 1

                    logged_once.add(lbl)
                    last_logged_at[lbl] = now_ts
                    consecutive_hits[lbl] = 0
                    status_message = f"Marked: {det.label.display_name} @ {now_iso}"
                    print(f"[live-attendance] {status_message}", file=sys.stderr)

                    if args.webhook_url:
                        post_webhook(args.webhook_url, event_row)

                    if args.save_snapshots:
                        snapshot_dir.mkdir(parents=True, exist_ok=True)
                        ts_token = now_iso.replace(":", "-").replace(".", "-")
                        cv2.imwrite(str(snapshot_dir / f"{ts_token}_{lbl}.jpg"), frame)

            if args.no_display:
                print(status_message, file=sys.stderr)
                continue

            overlay = draw_overlay(
                frame, latest_detections, len(profiles),
                len(session_attendance), source_label, status_message
            )
            cv2.imshow("Live Attendance", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    write_summary(summary_path, source_label, len(profiles), session_started_at, session_attendance)
    print(f"Attendance events: {event_log_path}")
    print(f"Session summary:   {summary_path}")
    print(f"Employees marked:  {len(session_attendance)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
