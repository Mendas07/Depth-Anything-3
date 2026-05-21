import cv2
import os
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# PATHS
# ============================================================

RGB_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "val"
    / "rgb"
)

VIS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "depth_color"
    / "depth_vis"
)

OUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "da3-val_spec-large"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEBUG PATHS
# ============================================================

print("\n==============================")
print("PATH CHECK")
print("==============================")

print(f"PROJECT_ROOT : {PROJECT_ROOT}")
print(f"RGB_DIR      : {RGB_DIR}")
print(f"VIS_DIR      : {VIS_DIR}")
print(f"OUT_DIR      : {OUT_DIR}")

print(f"\nRGB exists   : {RGB_DIR.exists()}")
print(f"VIS exists   : {VIS_DIR.exists()}")

print("==============================\n")


# ============================================================
# EXTENSÕES
# ============================================================

valid_exts = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".PNG",
    ".JPG",
    ".JPEG",
    ".BMP"
}


# ============================================================
# RGB IMAGES
# ============================================================

rgbs = sorted(
    [
        p for p in RGB_DIR.rglob("*")
        if p.suffix in valid_exts
    ],
    key=lambda x: x.name
)


# ============================================================
# DEPTH IMAGES
# ============================================================

depths = sorted(
    [
        p for p in VIS_DIR.rglob("*")
        if p.suffix in valid_exts
    ]
)


# ============================================================
# INFO
# ============================================================

print(f"Imagens de entrada : {len(rgbs)}")
print(f"Imagens geradas    : {len(depths)}")


# ============================================================
# CHECK
# ============================================================

if len(rgbs) != len(depths):

    print(
        f"\n⚠️ Aviso:"
        f"\nQuantidade diferente!"
        f"\nAlinhando apenas "
        f"{min(len(rgbs), len(depths))} imagens.\n"
    )


# ============================================================
# LOOP
# ============================================================

for idx, (r_path, d_path) in enumerate(
    zip(rgbs, depths)
):

    print(
        f"[{idx+1}/{min(len(rgbs), len(depths))}] "
        f"{r_path.name}"
    )

    img = cv2.imread(str(d_path))

    # ========================================================
    # OUTPUT NAME
    # ========================================================

    output_path = OUT_DIR / r_path.name

    cv2.imwrite(
        str(output_path),
        img
    )


# ============================================================
# FINAL
# ============================================================

print("\n===================================")
print("✅ Concluído!")
print("===================================\n")

print(f"Imagens salvas em:\n{OUT_DIR}\n")