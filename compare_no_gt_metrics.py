import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================
# CONFIGURAÇÃO DE CAMINHOS 
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parent

CSV_FINETUNED = PROJECT_ROOT / "outputs" / "metrics_fineepochs" / "no_gt_metrics_finetuned.csv"
CSV_FINEEPOCHS = PROJECT_ROOT / "outputs" / "metrics_fineepochs" / "finetunedep50.csv"
FIG_PATH = PROJECT_ROOT / "outputs" / "PLOT_MELHORIA1.png"

print(f"📂 Lendo métricas FINE-TUNED de: {CSV_FINETUNED}")
print(f"📂 Lendo métricas FINEEPOCHS de: {CSV_FINEEPOCHS}")

# ==========================
# VERIFICAÇÃO DE ARQUIVOS
# ==========================
if not CSV_FINETUNED.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {CSV_FINETUNED}")

if not CSV_FINEEPOCHS.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {CSV_FINEEPOCHS}")

# ==========================
# CARREGAMENTO DOS DADOS
# ==========================
df_finetuned = pd.read_csv(CSV_FINETUNED)
df_fineepochs = pd.read_csv(CSV_FINEEPOCHS)

# ==========================
# CÁLCULO DAS MÉDIAS
# ==========================
metrics = ["edge_alignment", "smoothness", "variance"]

means_finetuned = {m: df_finetuned[m].mean() for m in metrics}
means_fineepochs = {m: df_fineepochs[m].mean() for m in metrics}

# ==========================
# CÁLCULO DA MELHORA PERCENTUAL
# ==========================
improvements = {}
for m in metrics:
    orig = means_finetuned[m]
    fine = means_fineepochs[m]
    
    if abs(orig) < 1e-8:
        improvements[m] = 0.0
    else:
        improvements[m] = ((fine - orig) / orig) * 100

# ==========================
# IMPRESSÃO DETALHADA NO TERMINAL
# ==========================
print("\n================ COMPARAÇÃO DE MÉTRICAS (NO GT) ================\n")

for m in metrics:
    print(f"🔹 Métrica: {m}")
    print(f"   Fine-Tuned   : {means_finetuned[m]:.6f}")
    print(f"   Fineepochs : {means_fineepochs[m]:.6f}")
    print(f"   📈 Melhora percentual: {improvements[m]:.2f}%\n")

# ==========================
# PREPARAÇÃO DOS DADOS PARA O GRÁFICO
# ==========================
orig_vals = np.array([means_finetuned[m] for m in metrics])
fine_vals = np.array([means_fineepochs[m] for m in metrics])

# Normalização conjunta (escala justa entre métricas)
combined = np.vstack([orig_vals, fine_vals])
min_vals = combined.min(axis=0)
max_vals = combined.max(axis=0)
range_vals = (max_vals - min_vals) + 1e-8

orig_norm = (orig_vals - min_vals) / range_vals
fine_norm = (fine_vals - min_vals) / range_vals

# Labels explicativos 
labels = [
    "Edge Alignment\n(Maior = Melhor estrutura)",
    "Smoothness\n(Menor = Melhor suavidade)",
    "Variance\n(Detalhamento de profundidade)"
]

x = np.arange(len(labels))
width = 0.35

# ==========================
# CRIAÇÃO DO GRÁFICO MELHORADO
# ==========================
plt.figure(figsize=(14, 8))

bars1 = plt.bar(x - width/2, orig_norm, width, label="Original (Baseline)")
bars2 = plt.bar(x + width/2, fine_norm, width, label="Fine-Tuned")

# ==========================
# ANOTAÇÕES COMPLETAS NO GRÁFICO
# ==========================
for i, m in enumerate(metrics):
    improvement = improvements[m]
    
    # Texto de melhoria
    improvement_text = f"{improvement:+.1f}%"
    
    # Valores absolutos 
    abs_text = (
        f"Fine-Tuned: {orig_vals[i]:.4f}\n"
        f"Fineepochs: {fine_vals[i]:.4f}"
    )
    
    # Cor da melhoria (verde = melhor, vermelho = pior)
    if m == "smoothness":
        # Para smoothness, MENOR é melhor
        color = "green" if improvement < 0 else "red"
    else:
        # Para edge e variance, MAIOR é melhor
        color = "green" if improvement > 0 else "red"
    
    # Texto da melhoria percentual (acima)
    plt.text(
        x[i],
        max(orig_norm[i], fine_norm[i]) + 0.05,
        improvement_text,
        ha="center",
        fontsize=13,
        fontweight="bold",
        color=color
    )
    
    # Texto dos valores absolutos (abaixo)
    plt.text(
        x[i],
        -0.22,
        abs_text,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", alpha=0.1)
    )

# ==========================
#      TÍTULO
# ==========================
plt.xticks(x, labels, fontsize=11)
plt.ylabel("Valor Normalizado (Comparação Justa entre Métricas)", fontsize=12)

plt.title(
    "Comparação de Métricas Sem Ground Truth (Underwater Depth Estimation)\n"
    "Depth Anything 3 — Baseline vs Fine-Tuned\n"
    "Topo: Melhora Percentual | Base: Valores Absolutos",
    fontsize=14,
    fontweight="bold"
)

plt.legend(fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.4)

# Espaço extra para textos inferiores
plt.ylim(-0.35, 1.25)

plt.tight_layout()
plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")

print(f"\n📊 Gráfico comparativo COMPLETO salvo em: {FIG_PATH}")
print("✅ Análise de melhora percentual concluída com sucesso!")