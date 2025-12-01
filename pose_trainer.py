# trainer_rewrite.py
"""
Trainer for quaternion regression with proper logging and GPU monitoring.
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
CSV_PATH = "dataset_csv/rotations_20251130_235509.csv"
IMG_DIR = "dataset"
BATCH_SIZE = 128
NUM_EPOCHS = 250
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_COSINE_QUAT_LOSS = True
PRINT_EVERY_BATCH = 20
SEED = 42
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

# ============== GPU USAGE HELPER ==============
def log_gpu_usage():
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated(DEVICE) / 1024**2
        mem_reserved = torch.cuda.memory_reserved(DEVICE) / 1024**2
        # optional: get GPU utilization via nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            util = result.stdout.strip()
        except Exception:
            util = "N/A"
        log.info(f"[GPU] Memory allocated: {mem_alloc:.1f} MiB, Memory reserved: {mem_reserved:.1f} MiB, Utilization: {util}%")

# ============== MODEL ==============
class PoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        
        # Replace adaptive avg pool with fixed 7x7 avg pool
        backbone.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        # Replace fully connected layers
        backbone.fc = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4)
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# ============== LOSSES ==============
class QuaternionCosineLoss(nn.Module):
    def forward(self, pred, target):
        dot = torch.sum(pred * target, dim=1)
        return (1.0 - torch.abs(dot)).mean()

class QuaternionMSELoss(nn.Module):
    def forward(self, pred, target):
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
        q = q / (torch.norm(q) + 1e-8)
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
    def __init__(self, gamma_min=0.7, gamma_max=1.5):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def __call__(self, img):
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return F.adjust_gamma(img, gamma)

# class RandomDirectionalShading(object):
#     def __init__(self, strength=0.4, probability=0.7):
#         self.strength = strength
#         self.probability = probability

#     def __call__(self, img):
#         if random.random() > self.probability:
#             return img

#         w, h = img.size
#         angle = random.uniform(0, 2 * np.pi)
#         dx, dy = np.cos(angle), np.sin(angle)

#         # Build a gradient mask
#         gradient = Image.new("L", (w, h))
#         for y in range(h):
#             for x in range(w):
#                 # Project pixel onto light direction axis
#                 v = (x * dx + y * dy) / (w + h)
#                 v = 128 + v * 255 * self.strength
#                 gradient.putpixel((x, y), int(np.clip(v, 0, 255)))

#         gradient_rgb = gradient.convert("RGB")
#         return Image.blend(img, gradient_rgb, 0.35)

# ============== TRANSFORMS ==============
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # --- photometric augmentation ---
    transforms.ColorJitter(
        brightness=0.6,
        contrast=0.6,
        saturation=0.4,
        hue=0.06
    ),

    RandomGamma(0.7, 1.6),          # exposure changes
    #RandomDirectionalShading(0.35), # directional lighting on the object ONLY

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ============== LOAD DATA ==============
df = pd.read_csv(CSV_PATH)
indices = list(range(len(df)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=SEED)

train_ds = Subset(PoseDataset(CSV_PATH, IMG_DIR, train_transform), train_idx)
val_ds = Subset(PoseDataset(CSV_PATH, IMG_DIR, val_transform), val_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ============== SETUP ==============
model = PoseModel().to(DEVICE)
criterion = QuaternionCosineLoss() if USE_COSINE_QUAT_LOSS else QuaternionMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
best_val = float("inf")
patience = 40
patience_counter = 0

log.info(f"Starting training for {NUM_EPOCHS} epochs...")

# ============== TRAIN LOOP ==============
for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    running = 0.0
    for i, (imgs, targets) in enumerate(train_loader):
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        preds = preds / (torch.norm(preds, dim=1, keepdim=True) + 1e-8)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        running += loss.item()
        
        if i % PRINT_EVERY_BATCH == 0:
            log.info(f"[Epoch {epoch}] Batch {i}/{len(train_loader)} loss={loss.item():.4f}")
            log_gpu_usage()

    avg_train = running / max(1, len(train_loader))

    # Validation
    model.eval()
    vloss = 0.0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            preds = model(imgs)
            preds = preds / (torch.norm(preds, dim=1, keepdim=True) + 1e-8)
            vloss += criterion(preds, targets).item()
    avg_val = vloss / max(1, len(val_loader))
    scheduler.step(avg_val)

    log.info(f"Epoch {epoch} summary -> train={avg_train:.4f} val={avg_val:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
    log_gpu_usage()

    if avg_val < best_val:
        best_val = avg_val
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join("checkpoints", "pose_model_best.pt"))
        log.info("Saved NEW BEST model")
    else:
        patience_counter += 1
        log.info(f"No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience:
            log.info("EARLY STOPPING")
            break

torch.save(model.state_dict(), os.path.join("checkpoints", "pose_model_final.pt"))
log.info(f"Training complete. Best val: {best_val}")
