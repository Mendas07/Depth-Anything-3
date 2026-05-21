from pathlib import Path
from datetime import datetime
from typing import List

import subprocess
import platform
import os
import sys

from depth_anything_3.services.inference_service import run_inference


# =====================================
# PATHS
# =====================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_dir = PROJECT_ROOT / "datasets" / "val" / "rgb"

output_dir = PROJECT_ROOT / "outputs" / "depth_color"
output_dir.mkdir(parents=True, exist_ok=True)

# =====================================
# LOGS
# =====================================

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / "batch_inference_log.txt"

# =====================================
# MODEL
# =====================================

model_name = "da3-large"

device = "cuda"

batch_size = 8

# =====================================
# LOG FILE
# =====================================

with open(log_file, "w") as log:

    log.write("=========================================\n")
    log.write(" Depth Anything 3 Batch Inference Log\n")
    log.write("=========================================\n\n")

    log.write(f"Date: {datetime.now()}\n\n")

    # =====================================
    # SYSTEM
    # =====================================

    log.write("===== SYSTEM =====\n")

    log.write(f"Platform: {platform.platform()}\n")
    log.write(f"Python Version: {platform.python_version()}\n")
    log.write(f"Python Executable: {sys.executable}\n\n")

    # =====================================
    # CONDA
    # =====================================

    log.write("===== CONDA =====\n")

    conda_env = os.environ.get(
        "CONDA_DEFAULT_ENV",
        "Not detected"
    )

    conda_prefix = os.environ.get(
        "CONDA_PREFIX",
        "Not detected"
    )

    log.write(f"Conda Environment: {conda_env}\n")
    log.write(f"Conda Path: {conda_prefix}\n\n")

    # =====================================
    # GPU INFO
    # =====================================

    log.write("===== GPU =====\n")

    try:

        gpu_info = subprocess.check_output(
            ["nvidia-smi"],
            stderr=subprocess.STDOUT
        ).decode()

        log.write(gpu_info + "\n")

    except:

        log.write("nvidia-smi not found\n\n")

    # =====================================
    # NVCC
    # =====================================

    log.write("===== CUDA / NVCC =====\n")

    try:

        nvcc_info = subprocess.check_output(
            ["nvcc", "-V"],
            stderr=subprocess.STDOUT
        ).decode()

        log.write(nvcc_info + "\n")

    except:

        log.write("NVCC not found\n\n")

    # =====================================
    # PIP PACKAGES
    # =====================================

    log.write("===== INSTALLED PACKAGES =====\n")

    try:

        pip_packages = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.STDOUT
        ).decode()

        log.write(pip_packages + "\n")

    except:

        log.write("Could not list pip packages\n\n")

    # =====================================
    # EXECUTION INFO
    # =====================================

    log.write("===== EXECUTION =====\n")

    log.write(f"Model: {model_name}\n")
    log.write(f"Device: {device}\n")
    log.write(f"Batch Size: {batch_size}\n")

    log.write(f"Input directory: {input_dir}\n")
    log.write(f"Output directory: {output_dir}\n\n")


# =====================================
# EXTENSÕES
# =====================================

extensions = [
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.bmp"
]

image_paths = []

for ext in extensions:

    image_paths.extend(
        sorted(input_dir.glob(ext))
    )

image_paths = sorted(
    image_paths,
    key=lambda x: x.name
)

print(f"\nTotal images found: {len(image_paths)}\n")

# salva no log
with open(log_file, "a") as log:

    log.write(
        f"Total images found: {len(image_paths)}\n\n"
    )

# =====================================
# CHUNKS
# =====================================

def chunks(lst, n):

    for i in range(0, len(lst), n):

        yield lst[i:i + n]

# =====================================
# LOOP PRINCIPAL
# =====================================

batches = list(
    chunks(
        image_paths,
        batch_size
    )
)

for idx, batch in enumerate(batches):

    print(
        f"\n[{idx+1}/{len(batches)}] "
        f"Processando batch..."
    )

    with open(log_file, "a") as log:

        log.write(
            "\n====================================\n"
        )

        log.write(
            f"Batch: {idx+1}/{len(batches)}\n"
        )

        log.write(
            f"Batch size: {len(batch)}\n\n"
        )

        for img_path in batch:

            log.write(
                f"Image: {img_path}\n"
            )

    try:

        run_inference(
            image_paths=[
                str(p) for p in batch
            ],

            export_dir=str(output_dir),

            model_dir=model_name,

            device=device,

            export_format="depth_vis"
        )

        print(
            f"[OK] Batch {idx+1}"
        )

        with open(log_file, "a") as log:

            log.write("[SUCCESS]\n")

    except Exception as e:

        print(
            f"[ERRO] Batch {idx+1}"
        )

        with open(log_file, "a") as log:

            log.write("[FAILED]\n")
            log.write(f"{str(e)}\n")

# =====================================
# FINAL
# =====================================

with open(log_file, "a") as log:

    log.write("\n====================================\n")
    log.write("Inference Finished\n")
    log.write("====================================\n")

print("\n===================================")
print("Inferência finalizada.")
print("===================================\n")

print(f"Log salvo em: {log_file}\n")