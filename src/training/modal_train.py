"""Modal cloud GPU training for contour segmentation U-Net.

This module provides cloud GPU training using Modal. The training logic
is imported from train.py, which can also be used for local training.

Usage:
    # One-time: Upload training data to Modal volume
    modal run src/training/modal_train.py::upload_data --local-path data/training/N60E014

    # Run training on cloud T4 GPU
    modal run src/training/modal_train.py::train_remote --epochs 100 --batch-size 8

    # Download results after training
    modal run src/training/modal_train.py::download_results --run-name run_20260128_150000

    # List available runs in the models volume
    modal run src/training/modal_train.py::list_runs
"""  # noqa: E501

from pathlib import Path

import modal

# Project root is three levels up from this file
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Modal app definition
app = modal.App("topovision-training")

# Persistent volumes for data and model outputs
data_volume = modal.Volume.from_name("topovision-data", create_if_missing=True)
models_volume = modal.Volume.from_name("topovision-models", create_if_missing=True)

# GPU training image with all dependencies via pip install
# Using pip directly instead of uv sync to avoid venv issues
training_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")  # OpenCV dependencies
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "segmentation-models-pytorch>=0.3.0",
        "albumentations>=1.3.0",
        "tensorboard>=2.12.0",
        "opencv-python-headless>=4.8.0",
        "numpy",
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(PROJECT_ROOT / "src", remote_path="/root/src")  # mounted at runtime
)


@app.function(
    image=training_image,
    gpu="T4",
    volumes={"/data": data_volume, "/models": models_volume},
    timeout=14400,  # 4 hours
)
def train_remote(
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    encoder: str = "resnet34",
    val_split: float = 0.15,
) -> str:
    """Run training on Modal cloud GPU.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        encoder: Encoder backbone (resnet18, resnet34, resnet50, etc.).
        val_split: Fraction of data for validation.

    Returns:
        Name of the run directory containing trained models.
    """
    from src.training.config import TrainingConfig
    from src.training.train import train

    # Configure paths to use Modal volumes
    config = TrainingConfig(
        data_dir=Path("/data"),
        output_dir=Path("/models"),
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=learning_rate,
        encoder_name=encoder,
        device="cuda",
        val_split=val_split,
    )

    print("=" * 60)
    print("Modal Cloud Training - T4 GPU")
    print("=" * 60)
    print("Data volume: /data")
    print("Models volume: /models")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Encoder: {encoder}")
    print("=" * 60)

    # Run training
    train(config)

    # Commit volume changes to persist models
    models_volume.commit()

    # Find the latest run directory
    import os

    runs = sorted(
        [d for d in os.listdir("/models") if d.startswith("run_")], reverse=True
    )
    if runs:
        latest_run = runs[0]
        print(f"\n✓ Training complete! Run saved as: {latest_run}")
        print(
            f"  Download with: modal run src/training/modal_train.py::download_results --run-name {latest_run}"  # noqa: E501
        )
        return latest_run
    return "unknown"


@app.function(
    image=training_image,
    volumes={"/data": data_volume},
    timeout=3600,  # 1 hour for upload
)
def count_data_files() -> int:
    """Count files in the Modal data volume (no upload performed)."""
    import os

    file_count = 0
    for _root, _dirs, files in os.walk("/data"):
        file_count += len(files)

    print(f"Data volume contains {file_count} files")
    return file_count


@app.local_entrypoint()
def upload_data_local(local_path: str = "data/training"):
    """Upload local training data to Modal volume (recursive).

    Uploads all PNG files from the specified directory and its subdirectories,
    preserving the directory structure.

    Args:
        local_path: Local path to training data directory.
    """
    local_dir = Path(local_path)
    if not local_dir.exists():
        print(f"Error: Local path does not exist: {local_dir}")
        return

    # Find all PNG files recursively
    png_files = list(local_dir.rglob("*.png"))
    print(f"Found {len(png_files)} PNG files in {local_dir}")

    if not png_files:
        print("No PNG files found. Exiting.")
        return

    # Upload files to volume, preserving directory structure
    print("Uploading to Modal volume 'topovision-data'...")

    with data_volume.batch_upload() as batch:
        for i, png_file in enumerate(png_files):
            # Preserve relative path structure
            rel_path = png_file.relative_to(local_dir)
            remote_path = f"/{rel_path}"
            batch.put_file(png_file, remote_path)

            # Progress update every 500 files
            if (i + 1) % 500 == 0:
                print(f"  Uploaded {i + 1}/{len(png_files)} files...")

    print(f"✓ Uploaded {len(png_files)} files to volume")

    # Verify upload
    count = count_data_files.remote()
    print(f"✓ Volume now contains {count} files")


@app.function(
    image=training_image,
    volumes={"/models": models_volume},
    timeout=600,
)
def list_runs() -> list[str]:
    """List all training runs in the models volume."""
    import os

    if not os.path.exists("/models"):
        print("Models volume is empty")
        return []

    runs = sorted(
        [d for d in os.listdir("/models") if d.startswith("run_")], reverse=True
    )

    if not runs:
        print("No training runs found in models volume")
        return []

    print("Available training runs:")
    for run in runs:
        run_path = Path("/models") / run
        files = list(run_path.glob("*.pt"))
        file_names = [f.name for f in files]
        print(f"  {run}: {', '.join(file_names)}")

    return runs


@app.function(
    image=training_image,
    volumes={"/models": models_volume},
    timeout=600,
)
def get_run_files(run_name: str) -> dict[str, bytes]:
    """Get all files from a training run.

    Args:
        run_name: Name of the run directory (e.g., run_20260128_150000).

    Returns:
        Dictionary mapping filename to file contents.
    """
    run_path = Path("/models") / run_name
    if not run_path.exists():
        raise ValueError(f"Run not found: {run_name}")

    files = {}

    # Get model files
    for pt_file in run_path.glob("*.pt"):
        files[pt_file.name] = pt_file.read_bytes()

    # Get tensorboard logs
    tb_dir = run_path / "tensorboard"
    if tb_dir.exists():
        for tb_file in tb_dir.rglob("*"):
            if tb_file.is_file():
                rel_path = tb_file.relative_to(run_path)
                files[str(rel_path)] = tb_file.read_bytes()

    return files


@app.local_entrypoint()
def download_results(run_name: str, output_dir: str = "models"):
    """Download training results from Modal volume to local directory.

    Args:
        run_name: Name of the run directory (e.g., run_20260128_150000).
        output_dir: Local directory to save results.
    """
    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading run '{run_name}' to {output_path}...")

    try:
        files = get_run_files.remote(run_name)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nAvailable runs:")
        list_runs.remote()
        return

    for filename, content in files.items():
        file_path = output_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        print(f"  Downloaded: {filename}")

    print(f"\n✓ Downloaded {len(files)} files to {output_path}")
    print("\nView TensorBoard logs:")
    print(f"  tensorboard --logdir {output_path}/tensorboard")


@app.local_entrypoint()
def train_cli(
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    encoder: str = "resnet34",
    val_split: float = 0.15,
):
    """CLI entrypoint for training.

    Usage:
        modal run src/training/modal_train.py --epochs 100 --batch-size 8
    """
    print("Starting Modal cloud training...")
    run_name = train_remote.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        encoder=encoder,
        val_split=val_split,
    )
    print(f"\n✓ Training complete! Run: {run_name}")
    print("\nDownload results with:")
    print(
        f"modal run src/training/modal_train.py::download_results --run-name {run_name}"
    )
