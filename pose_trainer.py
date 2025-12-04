"""
Optimized Trainer for quaternion regression - A5000 GPU
ONNX-compatible export
Expected CSV columns: x,y,z,w,filename
Saves: pose_model_best.pt and pose_model_final.pt
"""
import os
import random
import logging
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import subprocess
import torchvision.transforms.functional as F

# ============== CONFIG ==============
CSV_PATH = "dataset_csv/rotations_20251203_150653.csv"
IMG_DIR = "dataset"
BATCH_SIZE = 64  # A5000 can handle this easily
NUM_EPOCHS = 400
LR = 2e-3  # higher LR, cosine will decay it
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_COSINE_QUAT_LOSS = True
PRINT_EVERY_BATCH = 10
SEED = 42
os.makedirs("checkpoints", exist_ok=True)

# ============== LOGGING ==============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ============== GPU USAGE HELPER ==============
def log_gpu_usage():
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated(DEVICE) / 1024**2
        mem_reserved = torch.cuda.memory_reserved(DEVICE) / 1024**2
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            util = result.stdout.strip()
        except Exception:
            util = "N/A"
        log.info(f"[GPU] Alloc: {mem_alloc:.1f} MiB, Reserved: {mem_reserved:.1f} MiB, Util: {util}%")

# ============== MODEL ==============
class PoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        
        # Replace adaptive avg pool with fixed 7x7 avg pool (ONNX compatible)
        backbone.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        # Replace fully connected layers with ONNX-safe architecture
        backbone.fc = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        self.backbone = backbone

    def forward(self, x):
        q = self.backbone(x)
        q = q / torch.norm(q, dim=1, keepdim=True).clamp(min=1e-8)
        return q

# ============== LOSSES ==============
class QuaternionCosineLoss(nn.Module):
    def forward(self, pred, target):
        # Normalize both - ONNX safe operations
        pred_norm = torch.norm(pred, dim=1, keepdim=True).clamp(min=1e-8)
        target_norm = torch.norm(target, dim=1, keepdim=True).clamp(min=1e-8)
        pred = pred / pred_norm
        target = target / target_norm
        dot = torch.sum(pred * target, dim=1)
        return (1.0 - torch.abs(dot)).mean()

class QuaternionMSELoss(nn.Module):
    def forward(self, pred, target):
        pred_norm = torch.norm(pred, dim=1, keepdim=True).clamp(min=1e-8)
        target_norm = torch.norm(target, dim=1, keepdim=True).clamp(min=1e-8)
        pred = pred / pred_norm
        target = target / target_norm
        return nn.functional.mse_loss(pred, target)

# ============== DATASET ==============
class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        required = {"x","y","z","w","filename"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"CSV missing columns, need: {required}")
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q = torch.tensor([row["x"], row["y"], row["z"], row["w"]], dtype=torch.float32)
        q_norm = torch.norm(q).clamp(min=1e-8)
        q = q / q_norm
        path = os.path.join(self.image_dir, row["filename"]).replace(".png",".jpg")
        if not os.path.exists(path):
            basename = os.path.basename(row["filename"])
            alt = os.path.join(self.image_dir, basename)
            if os.path.exists(alt):
                path = alt
            else:
                raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path).convert("RGBA")
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255,255,255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, q

class RandomGamma(object):
    def __init__(self, gamma_min=0.6, gamma_max=1.8):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def __call__(self, img):
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return F.adjust_gamma(img, gamma)

# ============== AGGRESSIVE AUGMENTATION ==============
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    
    # Photometric augmentation - aggressive
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=1.0,
            contrast=1.0,
            saturation=0.8,
            hue=0.15
        )
    ], p=0.9),
    
    RandomGamma(0.5, 2.0),
    
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ], p=0.5),
    
    transforms.RandomApply([
        transforms.RandomInvert(p=0.3)
    ], p=0.2),
    
    transforms.RandomEqualize(p=0.3),
    
    transforms.RandomApply([
        transforms.RandomAutocontrast(p=0.5)
    ], p=0.3),
    
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def main():
    """Main training function"""
    # deterministic
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    log.info(f"Using device: {DEVICE}")
    
    # ============== LOAD DATA ==============
    df = pd.read_csv(CSV_PATH)
    indices = list(range(len(df)))
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=SEED)

    train_ds = Subset(PoseDataset(CSV_PATH, IMG_DIR, train_transform), train_idx)
    val_ds = Subset(PoseDataset(CSV_PATH, IMG_DIR, val_transform), val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=20, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    log.info(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    # ============== SETUP ==============
    model = PoseModel().to(DEVICE)
    criterion = QuaternionCosineLoss() if USE_COSINE_QUAT_LOSS else QuaternionMSELoss()

    # AdamW with higher LR, cosine decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    # Early stopping
    best_val = float("inf")
    patience = 50
    patience_counter = 0

    log.info(f"Starting training for {NUM_EPOCHS} epochs...")
    log.info(f"Model: ResNet18 | Batch: {BATCH_SIZE} | LR: {LR} | Augmentation: AGGRESSIVE")

    # ============== TRAIN LOOP ==============
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        running = 0.0
        for i, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            preds = preds / (torch.norm(preds, dim=1, keepdim=True).clamp(min=1e-8))
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item()
            
            if i % PRINT_EVERY_BATCH == 0:
                log.info(f"[Epoch {epoch}] Batch {i}/{len(train_loader)} loss={loss.item():.6f}")

        avg_train = running / max(1, len(train_loader))

        # Validation
        model.eval()
        vloss = 0.0
        with torch.no_grad():

            preds = model(torch.cat([imgs[:16] for imgs,_ in train_loader], dim=0))
            print("Output norm mean:", preds.norm(dim=1).mean().item())

            for imgs, targets in val_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                preds = model(imgs)
                preds = preds / (torch.norm(preds, dim=1, keepdim=True).clamp(min=1e-8))
                vloss += criterion(preds, targets).item()
        avg_val = vloss / max(1, len(val_loader))
        scheduler.step()

        log.info(f"Epoch {epoch} -> train={avg_train:.6f} val={avg_val:.6f} lr={optimizer.param_groups[0]['lr']:.2e}")
        log_gpu_usage()

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join("checkpoints", "pose_model_best.pt"))
            log.info(f"✓ NEW BEST model (val={best_val:.6f})")
        else:
            patience_counter += 1
            if patience_counter % 10 == 0:
                log.info(f"No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                log.info("EARLY STOPPING triggered")
                break

    torch.save(model.state_dict(), os.path.join("checkpoints", "pose_model_final.pt"))
    log.info(f"Training complete. Best val: {best_val:.6f}")
    log.info("Training complete. Use export_onnx.py to convert to ONNX separately.")

if __name__ == '__main__':
    main()