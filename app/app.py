import os
import uuid
import shutil
from pathlib import Path

import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "best.pt"))
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Helmet Detection API",
    description="YOLOv8 REST API for helmet detection",
    version="1.0.0",
)

try:
    model = YOLO(str(MODEL_PATH))
except Exception as exc:
    raise RuntimeError(
        f"Could not load YOLO model from '{MODEL_PATH}'. "
        "Set MODEL_PATH to a valid .pt file."
    ) from exc


class PredictionResponse(BaseModel):
    filename: str
    detections: list[dict]
    result_url: str


@app.get("/")
async def root():
    return {"message": "Helmet Detection API is running. Visit /docs for Swagger UI."}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH.name}


@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are supported.",
        )

    file_id = uuid.uuid4().hex[:8]
    input_path = UPLOAD_DIR / f"{file_id}.jpg"

    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = model.predict(
            source=str(input_path),
            save=False,
            conf=0.25,
            verbose=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc
    finally:
        await file.close()

    pred = results[0]
    detections: list[dict] = []

    for box in pred.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        detections.append(
            {
                "class_id": class_id,
                "label": model.names[class_id],
                "confidence": round(confidence, 3),
                "box": [x1, y1, x2, y2],
            }
        )

    result_filename = f"{file_id}_result.jpg"
    result_path = RESULTS_DIR / result_filename
    annotated_img = pred.plot()
    cv2.imwrite(str(result_path), annotated_img)

    return PredictionResponse(
        filename=file.filename or result_filename,
        detections=detections,
        result_url=f"/download/{result_filename}",
    )


@app.get("/download/{image_name}")
async def download_result(image_name: str):
    safe_name = Path(image_name).name
    path = RESULTS_DIR / safe_name

    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Result not found.")

    return FileResponse(
        path=str(path),
        media_type="image/jpeg",
        filename=safe_name,
    )
