# Python CCTV Face Recognition — Training & Live Attendance

## Quick start (OpenCV LBPH path)

This is the recommended path: no dlib, no cmake, works on Python 3.11+.

### 1 — Install

```powershell
cd python-ml
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-webcam.txt
```

### 2 — Prepare dataset

One folder per employee, at least **5 raw photos** each (more = better):

```
python-ml/
  dataset/
    EMP001/
      01.jpg
      02.jpg
      03.jpg
    EMP002/
      ...
```

### 3 — Augment (optional but strongly recommended)

Generates ~50 augmented copies per base image using gamma, noise, rotation,
blur, contrast, sharpen, and flip variations:

```powershell
py augment_dataset.py
```

This writes augmented images back into each employee's dataset folder.

### 4 — Train the LBPH model

```powershell
py opencv_lbph_train.py `
  --dataset dataset `
  --output output\opencv `
  --metadata metadata.example.csv `
  --augment-level 2
```

Outputs:
- `output/opencv/lbph-model.yml`
- `output/opencv/lbph-labels.json`
- `output/opencv/lbph-training-report.json`

**Augment levels:**
| Level | What it adds |
|-------|--------------|
| 0     | No augmentation (fastest, lowest accuracy) |
| 1     | Flip + 3 brightness levels |
| 2     | Full: gamma × 7, noise × 4, rotation × 6, contrast × 5, blur, sharpen (default) |

### 5 — Run live attendance

**Laptop webcam:**
```powershell
py live_attendance.py `
  --model output\opencv\lbph-model.yml `
  --labels output\opencv\lbph-labels.json `
  --source 0
```

**RTSP CCTV camera:**
```powershell
py live_attendance.py `
  --model output\opencv\lbph-model.yml `
  --labels output\opencv\lbph-labels.json `
  --source "rtsp://user:pass@camera-ip:554/stream" `
  --no-display
```

Key options:
| Flag | Default | What it does |
|------|---------|--------------|
| `--distance-threshold` | 65 | Lower = stricter matching |
| `--min-consecutive-detections` | 2 | Frames a face must appear before marking |
| `--repeat-after-seconds` | 0 | 0 = mark once per session |
| `--save-snapshots` | off | Save a JPEG on each attendance mark |
| `--webhook-url` | - | POST attendance JSON to this URL |
| `--no-display` | off | Headless mode (for RTSP servers) |
| `--reconnect-delay` | 3 | Seconds before RTSP reconnect attempt |

## How the web gate verification works

The **Gate Terminal** in the web app uses **burst verification**:
1. User taps RFID badge at the gate
2. Browser sends 10 frames to the Node.js server
3. Node.js forwards frames to `opencv_face_service.py` (persistent worker)
4. Python detects faces with multi-scale Haar cascade + NMS, equalises with CLAHE,
   and runs LBPH recognition
5. Confidence is computed via sigmoid: `1 / (1 + e^(k*(d - 0.6*threshold)))`
6. Movement direction (entry/exit) is inferred from face position change across frames
7. Result shows in the Gate Terminal with color-coded confidence bar

## Metadata CSV

```csv
folder_name,employeeCode,name,department,rfidUid,email,phone,isActive
EMP001,EMP001,Anita Sharma,Operations,04A1BC22,anita@corp.com,+91-9000000001,true
```

## Tuning tips

| Problem | Fix |
|---------|-----|
| Too many false positives | Lower `--distance-threshold` (e.g. 55) |
| Too many rejections | Raise `--distance-threshold` (e.g. 75) |
| Bad under dim lighting | Capture dataset photos under same lighting, use `--augment-level 2` |
| Slow on high-res feed | Lower `--resize-width` (e.g. 640) |
| RTSP keeps dropping | Increase `--reconnect-delay` |
