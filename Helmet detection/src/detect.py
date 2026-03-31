import os
import argparse
import random
import cv2
from ultralytics import YOLO
from utils import ensure_dir, bundle_light_inference


def run_inference(weights, source, n_samples=10):
    """
    Run inference with YOLOv8, save prediction overlays + light bundle.
    """
    model = YOLO(weights)

    # Run predictions with YOLO defaults (runs/detect/exp, exp2, …)
    preds = model.predict(
        source=source,
        save=True,
        project="runs/detect",   # YOLO default root
        name="exp",              
        imgsz=640,
        exist_ok=False,          # YOLO will auto-increment exp, exp2...
        conf=0.25
    )

    # The actual YOLO run dir (e.g. runs/detect/exp3)
    run_dir = preds[0].save_dir
    run_name = os.path.basename(run_dir)  

    # Save random overlays
    overlay_dir = os.path.join(run_dir, "pred_overlays")
    ensure_dir(overlay_dir)

    sampled_preds = random.sample(preds, min(n_samples, len(preds)))
    for pred in sampled_preds:
        img_overlay = pred.plot()
        out_name = os.path.splitext(os.path.basename(pred.path))[0] + "_overlay.jpg"
        cv2.imwrite(os.path.join(overlay_dir, out_name), img_overlay)

    print(f"{len(sampled_preds)} overlays saved to {overlay_dir}")

    # Package into a light bundle
    bundle_light_inference(weights, overlay_dir,train_results_dir, run_name, output_dir = "Light_Bundles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="model/yolov11/best.pt", help="Path to weights")
    parser.add_argument("--source", type=str, default="data/images/test", help="Folder with test images")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of overlays to save")
    args = parser.parse_args()

    run_inference(
        weights=args.weights,
        source=args.source,
        n_samples=args.n_samples
    )

