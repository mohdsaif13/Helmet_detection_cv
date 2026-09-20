import argparse
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils import ensure_dir, bundle_light_inference


def run_inference(weights: str, source: str, n_samples: int = 10):
    """Run YOLO inference, save overlays, and optionally create a light bundle."""
    model = YOLO(weights)

    preds = model.predict(
        source=source,
        save=True,
        project="runs/detect",
        name="exp",
        imgsz=640,
        exist_ok=False,
        conf=0.25,
        verbose=False,
    )

    run_dir = Path(preds[0].save_dir)
    run_name = run_dir.name

    overlay_dir = run_dir / "pred_overlays"
    ensure_dir(str(overlay_dir))

    sampled_preds = random.sample(preds, min(n_samples, len(preds)))
    for pred in sampled_preds:
        img_overlay = pred.plot()
        out_name = f"{Path(pred.path).stem}_overlay.jpg"
        cv2.imwrite(str(overlay_dir / out_name), img_overlay)

    print(f"{len(sampled_preds)} overlays saved to {overlay_dir}")

    # Create a lightweight bundle only when a valid training-results directory is available.
    train_results_dir = Path("train/helmet_yolov8")
    if train_results_dir.exists():
        bundle_light_inference(
            weights_dir=str(Path(weights).parent),
            overlays_dir=str(overlay_dir),
            train_results_dir=str(train_results_dir),
            exp_name=run_name,
            output_dir="Light_Bundles",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run helmet detection inference")
    parser.add_argument("--weights", type=str, default="model/best.pt")
    parser.add_argument("--source", type=str, default="data/images/test")
    parser.add_argument("--n-samples", type=int, default=10)
    args = parser.parse_args()

    run_inference(
        weights=args.weights,
        source=args.source,
        n_samples=args.n_samples,
    )
