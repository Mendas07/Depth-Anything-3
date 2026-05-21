import cv2
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# PATHS
# ============================================================

SRC_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "da3-suim-large"
)

DST_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "da3-suim-large_bw"
)

DST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEBUG
# ============================================================

print("\n==============================")
print("PATH CHECK")
print("==============================")

print(f"SRC_DIR : {SRC_DIR}")
print(f"DST_DIR : {DST_DIR}")

print(f"\nSRC exists : {SRC_DIR.exists()}")

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
# IMAGE LIST
# ============================================================

image_paths = sorted(
    [
        p for p in SRC_DIR.rglob("*")
        if p.suffix in valid_exts
    ],
    key=lambda x: x.name
)


# ============================================================
# INFO
# ============================================================

print(f"Total imagens: {len(image_paths)}\n")


# ============================================================
# LOOP
# ============================================================

for idx, img_path in enumerate(image_paths):

    print(
        f"[{idx+1}/{len(image_paths)}] "
        f"{img_path.name}"
    )

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    img_color = cv2.imread(str(img_path))

    if img_color is None:

        print(
            f"⚠️ Erro ao carregar: "
            f"{img_path.name}"
        )

        continue

    # ========================================================
    # RGB -> GRAY
    # ========================================================

    img_gray = cv2.cvtColor(
        img_color,
        cv2.COLOR_BGR2GRAY
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_path = DST_DIR / img_path.name

    cv2.imwrite(
        str(output_path),
        img_gray
    )


# ============================================================
# FINAL
# ============================================================

print("\n===================================")
print("✅ Conversão concluída")
print("===================================\n")

print(f"Imagens salvas em:\n{DST_DIR}\n")