# 🪖 YOLOv8 Helmet Detection

Real-time helmet detection using **YOLOv8, OpenCV, PyTorch and FastAPI** for safety-compliance applications.

The repository contains the complete workflow for dataset preparation, YOLOv8 training, image/video inference, webcam detection, REST API serving and Docker deployment.

## ✨ Highlights

- Real-time helmet / no-helmet detection
- YOLOv8 object-detection pipeline
- **6,000+ annotated images** used for model development
- **86% mAP** reported in the project evaluation
- Approximately **1.5 ms inference latency** reported for the established inference pipeline
- FastAPI REST endpoint for image prediction
- Docker-ready deployment
- Webcam inference support
- Reproducible local dataset configuration

## 🧰 Tech Stack

**Python · YOLOv8 · Ultralytics · PyTorch · OpenCV · FastAPI · Docker · NumPy · Pillow**

## 📁 Project Structure

```text
Helmet_detection_cv/
├── app/
│   └── app.py                 # FastAPI application
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml              # Local dataset configuration
├── model/
│   └── best.pt                # Trained model (Git LFS)
├── notebooks/
│   └── helmet-detection-yolov8s.ipynb
├── src/
│   ├── train.py               # Training pipeline
│   ├── detect.py              # Image/video inference
│   ├── utils.py               # Training utilities
│   └── webcam_feed.py         # Live webcam detection
├── Dockerfile
├── requirements.txt
└── readme.md
```

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/mohdsaif13/Helmet_detection_cv.git
cd Helmet_detection_cv
```

### 2. Install dependencies

Python **3.10** is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Dataset configuration

The repository expects the dataset under:

```text
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

The included `data/data.yaml` uses repository-relative paths:

```yaml
path: .
train: data/images/train
val: data/images/val
test: data/images/test

nc: 2
names:
  0: with_helmet
  1: without_helmet
```

This avoids machine-specific paths such as local Google Drive or Windows directories.

## 🧠 Train the Model

Train with the modular pipeline:

```bash
python src/train.py --data data/data.yaml --epochs 50 --img-size 640 --batch-size 16
```

The best checkpoint is copied into:

```text
model/yolov8/best.pt
```

## 🔍 Run Inference

Using the trained model:

```bash
python src/detect.py --weights model/best.pt --source data/images/test
```

Or use another YOLO checkpoint:

```bash
python src/detect.py --weights path/to/best.pt --source path/to/images
```

Prediction overlays are saved under the generated `runs/detect/` directory.

## 📷 Webcam Detection

The webcam script now uses a portable model path instead of a machine-specific Windows path.

Default:

```bash
python src/webcam_feed.py
```

Change the camera index when necessary:

```text
CAMERA_INDEX=1
```

On Windows PowerShell:

```powershell
$env:CAMERA_INDEX="1"
python src/webcam_feed.py
```

Optional recording:

```powershell
$env:SAVE_OUTPUT="true"
python src/webcam_feed.py
```

## 🌐 FastAPI

Start the API:

```bash
uvicorn app.app:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
GET /health
```

Prediction:

```text
POST /predict/
```

Example:

```bash
curl -X POST -F "file=@sample.jpg" http://127.0.0.1:8000/predict/
```

The API returns detected class labels, confidence scores, bounding boxes and the path of the annotated result.

## 🐳 Docker

Build:

```bash
docker build -t helmet-detector .
```

Run:

```bash
docker run --rm -p 8000:8000 helmet-detector
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The container expects the model at:

```text
/app/model/best.pt
```

You can override it with:

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=/app/model/best.pt \
  helmet-detector
```

## 🔧 Issues Fixed

This repository was cleaned up to remove several portability/runtime problems:

- Removed hard-coded Windows/OneDrive model paths from the API and webcam application.
- Made the model location configurable through `MODEL_PATH`.
- Replaced machine-specific Google Drive paths in `data/data.yaml` with repository-relative paths.
- Fixed the YOLO training argument from `batch_size` to Ultralytics' `batch` parameter.
- Fixed the training results handling and project-directory configuration.
- Fixed the inference script's undefined `train_results_dir` reference.
- Fixed the default inference checkpoint path.
- Removed the invalid `project_dir` argument passed to the training function.
- Removed Docker's development-only `--reload` mode.
- Added API `/health` endpoint.
- Added safer file handling for uploaded images.
- Added environment-variable support for webcam camera selection and optional recording.
- Updated documentation to match the actual repository structure.

## 📊 Reported Project Results

According to the project results documented for this portfolio:

| Metric | Result |
|---|---:|
| Annotated training images | 6,000+ |
| Reported mAP | 86% |
| Reported inference latency | ~1.5 ms |
| Reported accuracy improvement after validation | 12% |

These figures describe the project's reported evaluation results and should be reproduced with the same dataset, model, hardware and evaluation procedure before being treated as independently benchmarked results.

## 🔐 Git LFS

The trained `model/best.pt` file is stored through **Git LFS**.

After cloning, make sure Git LFS is installed and pull the model:

```bash
git lfs install
git lfs pull
```

If you do not need the pretrained model, you can train a new checkpoint using `src/train.py`.

## 👨‍💻 Author

**MD Saif Ali**

AI/ML Engineer focused on Machine Learning, Generative AI, RAG, NLP and Computer Vision.

- GitHub: https://github.com/mohdsaif13
- LinkedIn: https://www.linkedin.com/in/md-saif-ali-a3250825b/

## 📄 License

No license file is currently included in this repository.
