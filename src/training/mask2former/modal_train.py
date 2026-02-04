"""Modal cloud GPU training for Mask2Former instance segmentation.

This module provides cloud GPU training using Modal with A10G or L4 GPUs,
which have sufficient memory for Mask2Former models.

Usage:
    # One-time: Upload training data with instance masks to Modal volume
    modal run src/training/mask2former/modal_train.py::upload_data \
        --local-path data/training/N60E014/N60E014

    # Upload multiple directories
    modal run src/training/mask2former/modal_train.py::upload_data \
        --local-path data/training/N60E012/N60E012
    modal run src/training/mask2former/modal_train.py::upload_data \
        --local-path data/training/N60E013/N60E013

    # Run training on cloud A10G GPU with multiple data directories
    modal run src/training/mask2former/modal_train.py::train_remote \
        --epochs 50 --batch-size 4 --data-subdirs N60E012 N60E013 N60E014

    # Download results after training
    modal run src/training/mask2former/modal_train.py::download_results \
        --run-name mask2former_20260203_150000

    # List available runs in the models volume
    modal run src/training/mask2former/modal_train.py::list_runs
"""

from pathlib import Path

import modal

# Project root is four levels up from this file
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Modal app definition
app = modal.App("topovision-mask2former-training")

# Persistent volumes for data and model outputs
data_volume = modal.Volume.from_name("topovision-data", create_if_missing=True)
models_volume = modal.Volume.from_name("topovision-models", create_if_missing=True)

# GPU training image with all dependencies
# Using A10G (24GB) or L4 (24GB) for Mask2Former memory requirements
training_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")  # OpenCV dependencies
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "timm>=0.9.0",
        "accelerate>=0.25.0",
        "albumentations>=1.3.0",
        "tensorboard>=2.12.0",
        "opencv-python-headless>=4.8.0",
        "numpy",
        "tqdm",
    )
    .env({"PYTHONPATH": "/root", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(PROJECT_ROOT / "src", remote_path="/root/src")
)


@app.function(
    image=training_image,
    gpu="A10G",  # 24GB VRAM, sufficient for Mask2Former
    volumes={"/data": data_volume, "/models": models_volume},
    timeout=60 * 60 * 8,  # 8 hours
)
def train_remote(
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    val_split: float = 0.1,
    gradient_accumulation: int = 1,
    data_subdirs: str = "",
) -> str:
    """Run Mask2Former training on Modal cloud GPU.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        val_split: Fraction of data for validation.
        gradient_accumulation: Gradient accumulation steps.
        data_subdirs: List of subdirectories in /data volume containing training data.
                      If None, auto-discovers all directories with instance masks.

    Returns:
        Name of the run directory containing trained models.
    """
    from src.training.mask2former.config import Mask2FormerTrainingConfig
    from src.training.mask2former.train import train

    # Find training data directories
    data_root = Path("/data")
    data_dirs: list[Path] = []

    if data_subdirs:
        # Parse comma-separated list and find tile directories
        subdir_list = [s.strip() for s in data_subdirs.split(",") if s.strip()]
        for subdir in subdir_list:
            subdir_path = data_root / subdir
            # Check if instance masks are directly in this dir
            if subdir_path.exists() and list(subdir_path.glob("*_instance_mask.png")):
                data_dirs.append(subdir_path)
            # Check nested subdirectory (e.g., N60E012/N60E012/)
            elif subdir_path.exists():
                for nested in subdir_path.iterdir():
                    if nested.is_dir() and list(nested.glob("*_instance_mask.png")):
                        data_dirs.append(nested)
            else:
                print(f"Warning: Directory not found: {subdir_path}")
    else:
        # Auto-discover all directories with instance masks (recursive)
        # Use rglob to search all subdirectories
        found_dirs = set()
        for instance_mask in data_root.rglob("*_instance_mask.png"):
            found_dirs.add(instance_mask.parent)
        data_dirs = sorted(found_dirs)

    if not data_dirs:
        msg = (
            "No directories with instance masks found. "
            "Make sure to generate data with --instance-mask flag and upload it."
        )
        raise ValueError(msg)

    print(f"Training data directories ({len(data_dirs)}):")
    total_masks = 0
    for d in data_dirs:
        mask_count = len(list(d.glob("*_instance_mask.png")))
        total_masks += mask_count
        print(f"  - {d.name}: {mask_count} instance masks")
    print(f"Total: {total_masks} instance masks")

    # Setup config
    config = Mask2FormerTrainingConfig(
        output_dir=Path("/models/mask2former"),
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation,
    )

    # Run training
    best_model_path = train(
        config=config,
        data_dirs=data_dirs,
        val_split=val_split,
    )

    # Commit changes to volume
    models_volume.commit()

    # Return the run name (parent directory of best_model.pt)
    return best_model_path.parent.name


@app.local_entrypoint()
def upload_data(local_path: str, remote_subdir: str = ""):
    """Upload training data to Modal volume.

    Args:
        local_path: Local path to training data directory.
        remote_subdir: Subdirectory name in the volume.
    """
    import os

    local_dir = Path(local_path)
    if not local_dir.exists():
        print(f"Error: {local_dir} does not exist")
        return

    # Check for instance masks
    instance_masks = list(local_dir.glob("*_instance_mask.png"))
    if not instance_masks:
        print(
            f"Warning: No instance masks found in {local_dir}. "
            "Make sure to generate data with --instance-mask flag."
        )

    remote_path = Path("/data") / (remote_subdir or local_dir.name)

    print(f"Uploading {local_dir} to Modal volume at {remote_path}")

    # Count files
    files = list(local_dir.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    print(f"Found {file_count} files to upload")

    # Upload using volume put
    with data_volume.batch_upload() as batch:
        for root, _dirs, filenames in os.walk(local_dir):
            for filename in filenames:
                local_file = Path(root) / filename
                relative = local_file.relative_to(local_dir)
                remote_file = str(remote_path / relative)
                batch.put_file(str(local_file), remote_file)

    print(f"Upload complete: {file_count} files")
    print(f"Instance masks: {len(instance_masks)}")


@app.function(
    image=training_image,
    volumes={"/models": models_volume},
    timeout=600,
)
def get_run_files(run_name: str) -> dict[str, bytes]:
    """Get all files from a training run (runs remotely).

    Args:
        run_name: Name of the run directory.

    Returns:
        Dictionary mapping filename to file contents.
    """
    run_path = Path("/models/mask2former") / run_name
    if not run_path.exists():
        available = list(Path("/models/mask2former").glob("*"))
        raise ValueError(
            f"Run not found: {run_name}. Available: {[r.name for r in available]}"
        )

    files = {}

    # Get all files recursively
    for file_path in run_path.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(run_path)
            files[str(rel_path)] = file_path.read_bytes()

    return files


@app.local_entrypoint()
def download_results(run_name: str, local_dir: str = "models/mask2former"):
    """Download trained models from Modal volume.

    Args:
        run_name: Name of the training run to download.
        local_dir: Local directory to save results.
    """
    output_path = Path(local_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {run_name} to {output_path}...")

    try:
        files = get_run_files.remote(run_name)
    except ValueError as e:
        print(f"Error: {e}")
        return

    for filename, content in files.items():
        file_path = output_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        size_mb = len(content) / (1024 * 1024)
        print(f"  Downloaded: {filename} ({size_mb:.2f} MB)")

    print(f"\n✓ Downloaded {len(files)} files to {output_path}")


@app.function(
    image=training_image,
    volumes={"/models": models_volume},
    timeout=60,
)
def get_available_runs() -> list[str]:
    """List available training runs (runs remotely)."""
    models_path = Path("/models/mask2former")
    if not models_path.exists():
        return []
    return [d.name for d in sorted(models_path.iterdir()) if d.is_dir()]


@app.local_entrypoint()
def list_runs():
    """List available training runs in the models volume."""
    print("Available Mask2Former training runs:")
    print("-" * 40)

    try:
        runs = get_available_runs.remote()
        if not runs:
            print("  No runs found.")
        else:
            for run in runs:
                print(f"  {run}")
    except Exception as e:
        print(f"Error listing runs: {e}")


@app.function(
    image=training_image,
    volumes={"/data": data_volume},
    timeout=60,
)
def get_available_data() -> list[tuple[str, int]]:
    """List available training data (runs remotely)."""
    data_path = Path("/data")
    if not data_path.exists():
        return []

    results = []
    for entry in sorted(data_path.iterdir()):
        if entry.is_dir():
            instance_masks = list(entry.rglob("*_instance_mask.png"))
            results.append((entry.name, len(instance_masks)))
    return results


@app.local_entrypoint()
def list_data():
    """List available training data in the data volume."""
    print("Available training data:")
    print("-" * 40)

    try:
        data = get_available_data.remote()
        if not data:
            print("  No data found.")
        else:
            for name, count in data:
                print(f"  {name} ({count} instance masks)")
    except Exception as e:
        print(f"Error listing data: {e}")
