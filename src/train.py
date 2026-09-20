import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from utils import ensure_dir, plot_training_curves


def train(
    data_yaml: str,
    epochs: int = 50,
    img_size: int = 640,
    batch_size: int = 16,
    weights: str = "yolov8s.pt",
    project_dir: str = "train/helmet_yolov8",
):
    """Train YOLOv8 and copy the best checkpoint into model/yolov8/."""
    model = YOLO(weights)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        amp=True,
        patience=15,
        project=project_dir,
        name="exp",
        exist_ok=False,
    )

    exp_dir = Path(results.save_dir)
    print(f"Training complete. Results saved to {exp_dir}")

    results_csv = exp_dir / "results.csv"
    plot_training_curves(str(results_csv), str(exp_dir))

    best_weight = exp_dir / "weights" / "best.pt"
    target_dir = Path("model") / "yolov8"
    ensure_dir(str(target_dir))

    if best_weight.exists():
        shutil.copy2(best_weight, target_dir / "best.pt")
        print(f"Best weights copied to {target_dir / 'best.pt'}")
    else:
        print("Warning: best.pt not found, copy skipped.")

    return str(exp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 helmet detector")
    parser.add_argument("--data", type=str, default="data/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weights", type=str, default="yolov8s.pt")
    parser.add_argument("--project-dir", type=str, default="train/helmet_yolov8")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        weights=args.weights,
        project_dir=args.project_dir,
    )
