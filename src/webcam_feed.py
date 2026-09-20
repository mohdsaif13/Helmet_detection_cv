import os
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "best.pt"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
SAVE_OUTPUT = os.getenv("SAVE_OUTPUT", "false").lower() == "true"
RUN_SECONDS = float(os.getenv("RUN_SECONDS", "0"))

model = YOLO(str(MODEL_PATH))
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera index {CAMERA_INDEX}. "
        "Try CAMERA_INDEX=1 or CAMERA_INDEX=2."
    )

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps != fps:
    fps = 10.0

output_path = None
out = None

if SAVE_OUTPUT:
    output_dir = BASE_DIR / "video_feed"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"webcam_output_{timestamp}.avi"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height),
    )

start_time = time.time()

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from camera.")
            break

        results = model.predict(frame, conf=0.25, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLO Helmet Detection", annotated_frame)

        if out is not None:
            out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if RUN_SECONDS > 0 and (time.time() - start_time) >= RUN_SECONDS:
            break
finally:
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

if output_path:
    print(f"Saved output video to {output_path}")
