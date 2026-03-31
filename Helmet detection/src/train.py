import os
import argparse
import shutil
from ultralytics import YOLO
from utils import ensure_dir, plot_training_curves


def train(data_yaml, epochs=50, img_size=640, weights="yolov8s.pt"):
    """
    Train YOLOv8 on the helmet dataset.
    Saves: weights, results.csv, plots.
    Also copies best.pt into model/yolov8/.
    """

   

    model = YOLO(weights)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch_size = 16,
        amp = True,
        patience = 15,
        imgsz=img_size,
        exist_ok=False
    )

    exp_dir = results[0].save_dir  # e.g. train/helmet_yolov8/exp
    print(f" Training complete. Results saved to {exp_dir}")

    # Plot training curves
    results_csv = os.path.join(exp_dir, "results.csv")
    plot_training_curves(results_csv, exp_dir)

    # Copy best weights to model/yolov8/
    best_weight = os.path.join(exp_dir, "weights", "best.pt")
    target_dir = "model/yolov8"
    ensure_dir(target_dir)
    if os.path.exists(best_weight):
        shutil.copy(best_weight, os.path.join(target_dir, "best.pt"))
        print(f" Best weights copied to {target_dir}/best.pt")
    else:
        print(" Warning: best.pt not found, copy skipped.")

    return exp_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--img", type=int, default=640, help="Image size for training")
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="Pretrained weights")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        epochs=args.epochs,
        img_size=args.img,
        weights=args.weights,
        project_dir="train/helmet_yolov8"
    )
