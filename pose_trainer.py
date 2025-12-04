# trainer_windows_safe_efficientnet.py
"""
Windows-safe trainer for quaternion regression with EfficientNet-Lite0:
- Gradient accumulation for large effective batch size
- Mixed precision training (AMP)
- Proper __main__ guard
- GPU usage logging
"""

import os
import random
import logging
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as F
import timm  # For EfficientNet-Lite

# ============== CONFIG ==============
CSV_PATH = "dataset_csv/rotations_20251203_194918.csv"
IMG_DIR = "dataset"
BATCH_SIZE = 64
EFFECTIVE_BATCH = 352
ACCUM_STEPS = EFFECTIVE_BATCH // BATCH_SIZE
NUM_EPOCHS = 350
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
USE_COSINE_QUAT_LOSS = True
PRINT_EVERY_BATCH = 20
os.makedirs("checkpoints", exist_ok=True)

# deterministic
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============== LOGGING ==============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)
log.info(f"Using device: {DEVICE}")

def log_gpu_usage():
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated(DEVICE)/1024**2
        mem_reserved = torch.cuda.memory_reserved(DEVICE)/1024**2
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            util = result.stdout.strip()
        except Exception:
            util = "N/A"
        log.info(f"[GPU] Memory allocated: {mem_alloc:.1f} MiB, reserved: {mem_reserved:.1f} MiB, Util: {util}%")

# ============== MODEL ==============
class PoseModelEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        # EfficientNet-Lite0 backbone (pretrained)
        self.backbone = timm.create_model('efficientnet_lite0', pretrained=True)
        in_features = self.backbone.classifier.in_features

        # Replace classifier with quaternion regression head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),  # quaternion output
        )

    def forward(self, x):
        x = self.backbone(x)
        # Normalize quaternion to unit length
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        return x

# ============== LOSSES ==============
class QuaternionCosineLoss(nn.Module):
    def forward(self, pred, target):
        dot = torch.sum(pred * target, dim=1)
        return (1.0 - torch.abs(dot)).mean()

class QuaternionMSELoss(nn.Module):
    def forward(self, pred, target):
        return nn.functional.mse_loss(pred, target)

# ============== DATASET ==============
class PoseDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, augment_quat=True):
        self.df = pd.read_csv(csv_file)
        required = {"x","y","z","w","filename"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"CSV missing columns, need: {required}")
        self.image_dir = image_dir
        self.transform = transform
        self.augment_quat = augment_quat

    def __len__(self):
        return len(self.df)

    def random_quat_perturb(self, q, max_angle_deg=5):
        angle = np.radians(random.uniform(-max_angle_deg, max_angle_deg))
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        sin_a = np.sin(angle/2)
        dq = np.array([axis[0]*sin_a, axis[1]*sin_a, axis[2]*sin_a, np.cos(angle/2)], dtype=np.float32)
        x1, y1, z1, w1 = q.numpy()
        x2, y2, z2, w2 = dq
        q_new = np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ], dtype=np.float32)
        q_new /= np.linalg.norm(q_new)
        return torch.tensor(q_new, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q = torch.tensor([row["x"], row["y"], row["z"], row["w"]], dtype=torch.float32)
        q = q / (torch.norm(q)+1e-8)
        if self.augment_quat:
            q = self.random_quat_perturb(q)

        path = os.path.join(self.image_dir, row["filename"]).replace(".png",".jpg")
        if not os.path.exists(path):
            alt = os.path.join(self.image_dir, os.path.basename(row["filename"]))
            if os.path.exists(alt):
                path = alt
            else:
                raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path).convert("RGBA")
        if img.mode=="RGBA":
            bg = Image.new("RGB", img.size, (255,255,255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, q

class RandomGamma:
    def __init__(self, gamma_min=0.7, gamma_max=1.5):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
    def __call__(self, img):
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return F.adjust_gamma(img, gamma)

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ColorJitter(0.6,0.6,0.4,0.06),
    RandomGamma(0.7,1.6),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ============== TRAIN FUNCTION ==============
def train():
    train_ds = PoseDataset(CSV_PATH, IMG_DIR, train_transform, augment_quat=True)
    val_ds = PoseDataset(CSV_PATH, IMG_DIR, val_transform, augment_quat=False)

    weights = np.ones(len(train_ds))
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = PoseModelEfficientNet().to(DEVICE)
    criterion = QuaternionCosineLoss() if USE_COSINE_QUAT_LOSS else QuaternionMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, min_lr=1e-6
    )

    scaler = torch.cuda.amp.GradScaler()
    best_val = float("inf")
    patience = 40
    patience_counter = 0

    log.info(f"Starting training for {NUM_EPOCHS} epochs...")

    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        running = 0.0
        optimizer.zero_grad()

        for i, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)

            with torch.cuda.amp.autocast():
                preds = model(imgs)
                loss = criterion(preds, targets)
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()
            running += loss.item() * ACCUM_STEPS

            if (i+1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if i % PRINT_EVERY_BATCH == 0:
                log.info(f"[Epoch {epoch}] Batch {i}/{len(train_loader)} loss={loss.item()*ACCUM_STEPS:.4f}")
                log_gpu_usage()

        avg_train = running / max(1, len(train_loader))

        # Validation
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                with torch.cuda.amp.autocast():
                    preds = model(imgs)
                    vloss += criterion(preds, targets).item()
        avg_val = vloss / max(1, len(val_loader))
        scheduler.step(avg_val)

        log.info(f"Epoch {epoch} -> train={avg_train:.4f} val={avg_val:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
        log_gpu_usage()

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join("checkpoints","pose_model_best.pt"))
            log.info("Saved NEW BEST model")
        else:
            patience_counter += 1
            log.info(f"No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                log.info("EARLY STOPPING")
                break

    torch.save(model.state_dict(), os.path.join("checkpoints","pose_model_final.pt"))
    log.info(f"Training complete. Best val: {best_val}")

# ============== ENTRY POINT ==============
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Windows-safe
    train()
