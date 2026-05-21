import torch
import os
from depth_anything_3.api import DepthAnything3
from pathlib import Path
from torchvision import transforms
from PIL import Image

# =========================
# CONFIGURAÇÕES
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 Escolher MELHOR modelo 
CHECKPOINT = "outputs/checkpoints/da3_finetuned_epoch_30.pth"

INPUT_DIR = "datasets/val/rgb"
OUTPUT_DIR = "outputs/finetuned_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
])

def test():

    print("⏳ Carregando arquitetura DA3-Large...")
    api_model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE").to(DEVICE)
    model = api_model.model.to(DEVICE)

    del api_model

    # Correção DINOv2
    if hasattr(model, "backbone") and hasattr(model.backbone, "pretrained"):
        if hasattr(model.backbone.pretrained, "export_feat_layers"):
            model.backbone.pretrained.export_feat_layers = [11, 15, 19, 23]

    # =========================
    # CARREGA PESOS 
    # =========================
    print(f"🔄 Carregando melhor modelo de: {CHECKPOINT}")
    state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)

    model.eval()

    image_paths = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    print(f"📂 Total de imagens encontradas: {len(image_paths)}")
    print("🚀 Iniciando inferência com modelo fine-tuned ...")

    with torch.no_grad():
        for img_path in image_paths:

            img_name = Path(img_path).stem
            image = Image.open(img_path).convert("RGB")

            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)
            image_tensor = image_tensor.unsqueeze(1).to(DEVICE)  # (1, 1, C, H, W)

            outputs = model(image_tensor)

            # Compatibilidade segura
            if hasattr(outputs, "depth"):
                depth = outputs.depth
            else:
                depth = outputs

            depth = depth.squeeze().cpu().numpy()

            # Normalização visual
            depth_min = depth.min()
            depth_max = depth.max()

            depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
            depth_img = (depth_norm * 255).astype("uint8")

            out_path = os.path.join(OUTPUT_DIR, f"{img_name}_depth.png")
            Image.fromarray(depth_img).save(out_path)

    print("\n✅ Inferência finalizada com sucesso!")
    print(f"📁 Resultados salvos em: {OUTPUT_DIR}")

if __name__ == "__main__":
    test()