"""
Fixed Quaternion Regression Trainer - A5000 GPU
Handles 23k+ images, no image rotation (rotations in filenames)
ONNX-compatible export
Expected CSV columns: x,y,z,w,filename
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

# ============== CONFIG ==============
CSV_PATH = "dataset_csv/rotations_20251203_150653.csv"
IMG_DIR = "dataset"
BATCH_SIZE = 32  # More conservative for better gradient stability
NUM_EPOCHS = 500
LR = 1e-3  # Lower initial LR for stability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY_BATCH = 20
SEED = 42
os.makedirs("checkpoints", exist_ok=True)

# Deterministic
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

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
        mem_alloc = torch.cuda.memory_allocated(DEVICE) / 1024**2
        mem_reserved = torch.cuda.memory_reserved(DEVICE) / 1024**2
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
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
        backbone.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        # Larger FC head for better quaternion regression
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)  # Output: x,y,z,w
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# ============== LOSS FUNCTIONS ==============
class QuaternionGeometricLoss(nn.Module):
    """Geodesic distance on quaternion manifold - most direct for rotation"""
    def forward(self, pred, target):
        # Normalize
        pred = pred / (torch.norm(pred, dim=1, keepdim=True).clamp(min=1e-8))
        target = target / (torch.norm(target, dim=1, keepdim=True).clamp(min=1e-8))
        
        # Compute dot product
        dot = torch.sum(pred * target, dim=1).clamp(-1.0, 1.0)
        
        # Geodesic distance: arccos(|dot|)
        # Use abs(dot) to handle quaternion double cover
        angle = torch.acos(torch.abs(dot) + 1e-8)
        
        return angle.mean()

class CombinedQuaternionLoss(nn.Module):
    """Combines geodesic + MSE for stability"""
    def __init__(self, w_geo=0.7, w_mse=0.3):
        super().__init__()
        self.w_geo = w_geo
        self.w_mse = w_mse
        
    def forward(self, pred, target):
        pred = pred / (torch.norm(pred, dim=1, keepdim=True).clamp(min=1e-8))
        target = target / (torch.norm(target, dim=1, keepdim=True).clamp(min=1e-8))
        
        # Geodesic
        dot = torch.sum(pred * target, dim=1).clamp(-1.0, 1.0)
        geo_loss = torch.acos(torch.abs(dot) + 1e-8).mean()
        
        # MSE
        mse_loss = nn.functional.mse_loss(pred, target)
        
        return self.w_geo * geo_loss + self.w_mse * mse_loss

# ============== DATASET ==============
class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        required = {"x", "y", "z", "w", "filename"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"CSV missing columns, need: {required}")
        
        # Normalize quaternions in CSV
        quats = self.df[["x", "y", "z", "w"]].values
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        self.df[["x", "y", "z", "w"]] = quats / norms.clip(min=1e-8)
        
        self.image_dir = image_dir
        self.transform = transform
        self.missing_count = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q = torch.tensor([row["x"], row["y"], row["z"], row["w"]], dtype=torch.float32)
        
        # Find image file
        filename = row["filename"]
        path = os.path.join(self.image_dir, filename)
        
        # Try variations if needed
        if not os.path.exists(path):
            alternatives = [
                os.path.join(self.image_dir, os.path.basename(filename)),
                path.replace(".png", ".jpg"),
                path.replace(".jpg", ".png"),
            ]
            for alt in alternatives:
                if os.path.exists(alt):
                    path = alt
                    break
        
        if not os.path.exists(path):
            self.missing_count += 1
            # Return dummy - shouldn't happen often with large dataset
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, q

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            log.warning(f"Failed to load {path}: {e}")
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, q

        if self.transform:
            img = self.transform(img)
        
        return img, q

# ============== LIGHT AUGMENTATION (no rotation!) ==============
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    
    # Light photometric only - no spatial transforms
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1
        )
    ], p=0.7),
    
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    ], p=0.3),
    
    transforms.RandomEqualize(p=0.1),
    
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============== LOAD DATA ==============
df = pd.read_csv(CSV_PATH)
log.info(f"Loaded {len(df)} samples from CSV")

indices = list(range(len(df)))
train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=SEED)

train_ds = Subset(PoseDataset(CSV_PATH, IMG_DIR, train_transform), train_idx)
val_ds = Subset(PoseDataset(CSV_PATH, IMG_DIR, val_transform), val_idx)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, 
    num_workers=16, pin_memory=True, drop_last=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, 
    num_workers=8, pin_memory=True
)

log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

# ============== SETUP ==============
model = PoseModel().to(DEVICE)
criterion = CombinedQuaternionLoss(w_geo=0.7, w_mse=0.3)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=40, T_mult=1, eta_min=1e-6
)

best_val = float("inf")
patience = 40
patience_counter = 0

log.info(f"Starting training: {NUM_EPOCHS} epochs, ResNet18 + large FC head")
log.info(f"Loss: Geodesic + MSE | LR: {LR} | Light augmentation (no rotation)")

# ============== TRAINING LOOP ==============
def train():
    global best_val, patience_counter
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for i, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(imgs)
            
            # Normalize predictions
            preds = preds / (torch.norm(preds, dim=1, keepdim=True).clamp(min=1e-8))
            
            loss = criterion(preds, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % PRINT_EVERY_BATCH == 0:
                log.info(f"[E{epoch}] Batch {i}/{len(train_loader)} loss={loss.item():.6f}")

            avg_train = running_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                    preds = model(imgs)
                    preds = preds / (torch.norm(preds, dim=1, keepdim=True).clamp(min=1e-8))
                    val_loss += criterion(preds, targets).item()
            
            avg_val = val_loss / len(val_loader)
            scheduler.step()

            log.info(f"Epoch {epoch} | train_loss={avg_train:.6f} | val_loss={avg_val:.6f} | lr={optimizer.param_groups[0]['lr']:.2e}")
            log_gpu_usage()

            if avg_val < best_val:
                best_val = avg_val
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join("checkpoints", "pose_model_best.pt"))
                log.info(f"✓ NEW BEST (val={best_val:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log.info("Early stopping triggered")
                    

        torch.save(model.state_dict(), os.path.join("checkpoints", "pose_model_final.pt"))
        log.info(f"Training complete. Best val: {best_val:.6f}")
        log.info("Use export_onnx.py to convert to ONNX separately.")

if __name__ == "__main__":
    train()