import numpy as np
import pandas as pd
import cv2
import os
from pathlib import Path

# ==========================
# CONFIG - RGB Original
# ==========================
BASE_PATH = Path("/home/pdi5060ti/Sorriso1909/depth-anything-3")

datasets_config = {
    "SUIM": {
        "rgb_ref": BASE_PATH / "suim",  
        "models": {
            "Small": BASE_PATH / "datasets/da3-suim-small_bw",
            "Base":  BASE_PATH / "datasets/da3-suim-base_bw",
            "Large": BASE_PATH / "datasets/da3-suim-large_bw",
        }
    },
    "USIS10K": {
        "rgb_ref": BASE_PATH / "usis10k", 
        "models": {
            "Small": BASE_PATH / "datasets/da3-usis10k-small_bw",
            "Base":  BASE_PATH / "datasets/da3-usis10k-base_bw",
            "Large": BASE_PATH / "datasets/da3-usis10k-large_bw",
        }
    },
    "UIEB": {
        "rgb_ref": BASE_PATH / "uieb",    
        "models": {
            "Small": BASE_PATH / "datasets/depth da3-small_bw", 
            "Base":  BASE_PATH / "datasets/depth da3-base_bw",
            "Large": BASE_PATH / "datasets/depth da3-large_bw",
        }
    }
}
# Pasta de saída para o CSV
OUT_DIR = BASE_PATH / "outputs/metrics_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "no_ref_metrics_da3.csv"

# ==========================
# MÉTRICAS
# ==========================

def edge_alignment_score(rgb, depth):
    # Converte RGB para cinza e detecta bordas
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges_rgb = cv2.Canny(rgb_gray, 50, 150)

    # Gradientes do mapa de profundidade
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    if grad_mag.max() < 1e-6:
        return 0.0

    grad_mag = grad_mag / grad_mag.max()
    edges_depth = (grad_mag > 0.1)

    intersection = np.logical_and(edges_rgb > 0, edges_depth).sum()
    union = np.logical_or(edges_rgb > 0, edges_depth).sum()

    return intersection / (union + 1e-8)

def depth_smoothness(depth):
    dx = np.abs(np.diff(depth, axis=1))
    dy = np.abs(np.diff(depth, axis=0))
    return float((dx.mean() + dy.mean()) / 2.0)

def edge_aware_smoothness(rgb, depth):
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    dx_depth = np.abs(np.diff(depth, axis=1))
    dy_depth = np.abs(np.diff(depth, axis=0))
    dx_img = np.abs(np.diff(rgb_gray, axis=1))
    dy_img = np.abs(np.diff(rgb_gray, axis=0))

    weight_x = np.exp(-dx_img)
    weight_y = np.exp(-dy_img)
    return float((dx_depth * weight_x).mean() + (dy_depth * weight_y).mean()) / 2.0

# ==========================
# LOOP DE PROCESSAMENTO
# ==========================

results = []

for ds_name, ds in datasets_config.items():
    # Busca arquivos de imagem na pasta de referência
    img_files = sorted(list(ds["rgb_ref"].glob("*.png")) + list(ds["rgb_ref"].glob("*.jpg")))
    
    if not img_files:
        print(f"⚠️ Nenhuma imagem encontrada para o dataset {ds_name}")
        continue

    print(f"\n📂 Dataset: {ds_name} | {len(img_files)} imagens")

    for model_name, depth_dir in ds["models"].items():
        metrics_stack = []

        for idx, img_path in enumerate(img_files):
            # No DA3 as predições costumam ter o mesmo nome do input
            depth_file = depth_dir / img_path.name
            
            if not depth_file.exists():
                continue

            # Carrega a "RGB" 
            rgb = cv2.imread(str(img_path))
            if rgb is None: continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # Carrega a Predição (Depth) do DA3
            depth_img = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE)
            if depth_img is None: continue
            depth = depth_img.astype(np.float32) / 255.0 # Normaliza 0-1

            # Garante que ambos tenham o mesmo tamanho
            if rgb.shape[:2] != depth.shape[:2]:
                depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))

            # Cálculo
            e = edge_alignment_score(rgb, depth)
            s = depth_smoothness(depth)
            eas = edge_aware_smoothness(rgb, depth)

            metrics_stack.append([e, s, eas])

        if metrics_stack:
            avg = np.mean(metrics_stack, axis=0)
            results.append([ds_name, model_name, avg[0], avg[1], avg[2]])
            print(f"   ✅ {model_name}: Processado.")

# ==========================
# GERAÇÃO DO DATAFRAME
# ==========================

df = pd.DataFrame(results, columns=[
    "Dataset", "Model", "Edge Align (↑)", "Smoothness (↓)", "EdgeAwareSmooth (↓)"
])

print("\n" + "="*80)
print("📊 RESULTADOS: MÉTRICAS SEM GROUND TRUTH (DA3)")
print("="*80)
print(df.to_string(index=False))
print("="*80)

df.to_csv(OUT_PATH, index=False)
print(f"\n💾 CSV salvo em: {OUT_PATH}")