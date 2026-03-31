import os
import shutil
import zipfile
import datetime
import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: str):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)



def plot_training_curves(csv_path, output_dir):
    """
    Reads results.csv and plots training curves.
    Saves two plots: loss metrics & validation metrics.
    """
    if not os.path.exists(csv_path):
        print(f"[Warning] {csv_path} not found. Skipping plots.")
        return

    df = pd.read_csv(csv_path)

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Loss metrics ---
    plt.figure()
    if "train/box_loss" in df.columns:
        plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
    if "train/cls_loss" in df.columns:
        plt.plot(df["epoch"], df["train/cls_loss"], label="Train Cls Loss")
    if "train/dfl_loss" in df.columns:
        plt.plot(df["epoch"], df["train/dfl_loss"], label="Train DFL Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Metrics")
    plt.legend()
    loss_plot_path = os.path.join(output_dir, "loss_metrics.png")
    plt.savefig(loss_plot_path)
    plt.close()

    # --- Validation metrics ---
    plt.figure()
    if "metrics/precision(B)" in df.columns:
        plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
    if "metrics/recall(B)" in df.columns:
        plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
    if "metrics/mAP50(B)" in df.columns:
        plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@50")
    if "metrics/mAP50-95(B)" in df.columns:
        plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@50-95")

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Metrics")
    plt.legend()
    val_plot_path = os.path.join(output_dir, "val_metrics.png")
    plt.savefig(val_plot_path)
    plt.close()

    print(f"[Info] Training plots saved to {output_dir}")


def bundle_light_inference(weights_dir, overlays_dir, train_results_dir, exp_name, output_dir="light_bundles"):
    """
    Creates a lightweight bundle with:
    - best.pt, last.pt
    - prediction overlays
    - training results.csv
    - loss_metrics.png, val_metrics.png
    - readme.txt
    """

    bundle_dir = os.path.join(output_dir, f"{exp_name}_bundle")
    os.makedirs(bundle_dir, exist_ok=True)


    # --- Copy weights ---
    for fname in ["best.pt", "last.pt"]:
        src = os.path.join(weights_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, bundle_dir)

    # --- Copy prediction overlays ---
    if os.path.exists(overlays_dir):
        dest_overlays = os.path.join(bundle_dir, "pred_overlays")
        shutil.copytree(overlays_dir, dest_overlays)

    # --- Copy training results.csv & plots ---
    results_csv = os.path.join(train_results_dir, "results.csv")
    if os.path.exists(results_csv):
        shutil.copy(results_csv, bundle_dir)

        # Generate training plots inside bundle
        plot_training_curves(results_csv, bundle_dir)

    # --- Create readme.txt ---
    readme_path = os.path.join(bundle_dir, "readme.txt")
    with open(readme_path, "w") as f:
        f.write(f"Weights: best.pt, last.pt\n")
        f.write(f"Predictions from: {overlays_dir}\n")
        f.write(f"Training results from: {train_results_dir}\n")

    # --- Zip it ---
    zip_path = os.path.join(output_dir, f"{exp_name}_bundle.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, _, files in os.walk(bundle_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, bundle_dir)
                zipf.write(file_path, arcname)

    print(f"[Info] Light bundle created: {zip_path}")
    return zip_path

