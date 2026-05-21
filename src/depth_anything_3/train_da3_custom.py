import torch
import os
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ==========================
# 1. AJUSTE DE CAMINHOS
# ==========================
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("❌ Erro: Não foi possível localizar o módulo 'depth_anything_3'.")
    print(f"Caminho atual no sys.path: {sys.path[-1]}")
    sys.exit(1)

# ==========================
# 2. DATASET CUSTOMIZADO
# ==========================
class PseudoDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.rgb_dir = os.path.join(self.root_dir, "rgb")
        self.depth_dir = os.path.join(self.root_dir, "depth")

        if not os.path.exists(self.rgb_dir):
            raise FileNotFoundError(f"Pasta não encontrada: {self.rgb_dir}")

        self.filenames = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith(('.png', '.jpg'))])
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        rgb_path = os.path.join(self.rgb_dir, img_name)
        depth_path = os.path.join(self.depth_dir, img_name)

        image = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")

        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)

        return image, depth

# ==========================
# 3. CONFIGURAÇÕES
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2  # ViT-L em GPUs de 8GB
LR = 1e-6       # Learning rate baixa para Fine-Tuning
EPOCHS = 30     # Evita overfitting nos pseudo-labels
DATASET_ROOT = "/home/pdi5060ti/Sorriso1909/depth-anything-3/datasets"
CHECKPOINT_DIR = "/home/pdi5060ti/Sorriso1909/depth-anything-3/outputs/checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Transformação padrão do Depth Anything 3
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
])

# ==========================
# 4. LOOP DE TREINO
# ==========================
def main():
    print("⏳ Carregando modelo DA3-Large...")
    model_api = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE").to(DEVICE)
    model = model_api.model.to(DEVICE)

    del model_api

    if hasattr(model.backbone, "pretrained"):
        if getattr(model.backbone.pretrained, "export_feat_layers", None) is None:
            model.backbone.pretrained.export_feat_layers = [11, 15, 19, 23]

    train_ds = PseudoDataset(DATASET_ROOT, split='train', transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    print(f"🚀 Iniciando Fine-Tuning em: {DEVICE}")
    print(f"📈 Imagens de treino: {len(train_ds)}")

    for epoch in range(EPOCHS):
        model.train()

        epoch_loss = 0

        for i, (images, depths) in enumerate(train_loader):
            images, depths = images.to(DEVICE), depths.to(DEVICE)

            optimizer.zero_grad()

            # Depth Anything 3 espera input no formato (B, N, 3, H, W)
            images = images.unsqueeze(1)

            outputs = model(images)

            # O modelo retorna um objeto com atributo .depth
            if hasattr(outputs, "depth"):
                output = outputs.depth
            else:
                output = outputs

            # Ajustar para formato (B, 1, H, W)
            if output.dim() == 3:
                output = output.unsqueeze(1)

            # Ajuste de dimensão (modelo pode retornar [B, 1, H, W])
            if output.shape != depths.shape:
                output = torch.nn.functional.interpolate(output, size=(518, 518), mode='bilinear', align_corners=False)

            loss = criterion(output, depths)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # MONITOR DE MEMÓRIA NO PRIMEIRO LOTE
            if epoch == 0 and i == 0 and torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated(DEVICE) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(DEVICE) / 1024**2
                print(f"\n📊 [GPU MEMORY] Alocada: {mem_alloc:.2f} MB | Reservada: {mem_reserved:.2f} MB")
                print(f"💡 Se a memória estiver próxima do limite, mantenha o Batch Size em {BATCH_SIZE}.\n")

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Iter [{i}/{len(train_loader)}] Loss: {loss.item():.6f}")

        # Salvar checkpoint ao final da época
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"da3_finetuned_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Checkpoint salvo: {ckpt_path}")

if __name__ == "__main__":
    main()
