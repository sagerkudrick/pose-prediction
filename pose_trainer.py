"""
Coolify-safe trainer for quaternion regression:
- Downloads dataset/CSV if missing
- Saves checkpoints to persistent volume
- Exports ONNX model to persistent volume
- Gradient accumulation for large batch
- Mixed precision training (AMP)
- Windows-safe multiprocessing
- FIX: Quaternion ambiguity correction (flips predictions, not targets)
- FIX: Proper train/val split from single CSV
- FIX: Consistent quaternion normalization
"""
import os
import random
import logging
import subprocess
import zipfile
import requests
import numpy as np
import pandas as pd
from PIL import Image
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torchvision.transforms.functional as F

def safe_remove(path):
    if os.path.isfile(path):
        try:
            os.remove(path)
        except:
            print(f"Failed to remove file: {path}")
    elif os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except:
            print(f"Failed to remove directory: {path}")

# ================= CONFIG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 25
BATCH_SIZE = 256
EFFECTIVE_BATCH = 512
ACCUM_STEPS = EFFECTIVE_BATCH // BATCH_SIZE
NUM_EPOCHS = 300
LR = 1e-5
USE_COSINE_QUAT_LOSS = True
PRINT_EVERY_BATCH = 20
TRAIN_VAL_SPLIT = 0.8
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Persistent paths
IMG_DIR = os.path.join(BASE_DIR, "dataset/dataset")
CSV_DIR = os.path.join(BASE_DIR, "dataset_csv")
CSV_PATH = os.path.join(CSV_DIR, "rotations.csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Hugging Face dataset
# DATASET_ZIP_URL = "https://huggingface.co/datasets/SagerKudrick/EngineRotations/resolve/main/dataset.zip"
# DATASET_CSV = "https://huggingface.co/datasets/SagerKudrick/EngineRotations/resolve/main/rotations.csv"
DATASET_ZIP_URL = "https://huggingface.co/datasets/SagerKudrick/pose-prediction-v2/resolve/main/renders.zip"
DATASET_CSV = "https://huggingface.co/datasets/SagerKudrick/pose-prediction-v2/resolve/main/rotations_20251207_155953.csv"

# ================= SEED =================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ================= LOGGING =================
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
                capture_output=True, text=True
            )
            util = result.stdout.strip()
        except Exception:
            util = "N/A"
        log.info(f"[GPU] Memory allocated: {mem_alloc:.1f} MiB, reserved: {mem_reserved:.1f} MiB, Util: {util}%")

# ================= DATASET DOWNLOAD =================
def download_and_extract_dataset():
    imgs = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".png", ".jpg"))]
    if len(imgs) == 0:
        os.makedirs(IMG_DIR, exist_ok=True)
        zip_path = os.path.join(IMG_DIR, "dataset.zip")
        log.info("Dataset not found, downloading from Hugging Face...")
        with requests.get(DATASET_ZIP_URL, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(1024*1024):
                    f.write(chunk)
        log.info("Unzipping dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(IMG_DIR)
        log.info("Dataset ready.")
    
    if not os.path.exists(CSV_PATH):
        log.info("Downloading CSV...")
        with requests.get(DATASET_CSV, stream=True) as r:
            r.raise_for_status()
            with open(CSV_PATH, "wb") as f:
                for chunk in r.iter_content(1024*1024):
                    f.write(chunk)
        log.info("CSV ready.")

# ================= CUSTOM LAYERS =================
class HardSwishManual(nn.Module):
    def forward(self, x):
        return x * nn.functional.relu6(x + 3) / 6

class HardSigmoidManual(nn.Module):
    def forward(self, x):
        return nn.functional.relu6(x + 3) / 6

def replace_hard_ops(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Hardswish):
            module.add_module(name, HardSwishManual())
        elif isinstance(child, nn.Hardsigmoid):
            module.add_module(name, HardSigmoidManual())
        else:
            replace_hard_ops(child)

# ================= MODEL =================
class PoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        replace_hard_ops(backbone)
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            HardSwishManual(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            HardSwishManual(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# ================= LOSSES =================
class QuaternionCosineLoss(nn.Module):
    def forward(self, pred, target):
        dot = torch.sum(pred * target, dim=1)
        return (1.0 - dot).mean()

class QuaternionMSELoss(nn.Module):
    def forward(self, pred, target):
        return nn.functional.mse_loss(pred, target)

# ================= DATASET CLASS =================
class PoseDataset(Dataset):
    def __init__(self, csv_file, image_dir, indices=None, transform=None):
        """
        Args:
            csv_file: path to CSV
            image_dir: path to images
            indices: list of row indices to use (for train/val split)
            transform: augmentation pipeline
        """
        self.df = pd.read_csv(csv_file)
        required = {"x", "y", "z", "w", "filename"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"CSV missing columns, need: {required}")
        
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q = torch.tensor([row["x"], row["y"], row["z"], row["w"]], dtype=torch.float32)
        q = q / (torch.norm(q) + 1e-8)
        
        path = os.path.join(self.image_dir, row["filename"]).replace(".png", ".jpg")
        if not os.path.exists(path):
            alt = os.path.join(self.image_dir, os.path.basename(row["filename"]))
            if os.path.exists(alt):
                path = alt
            else:
                raise FileNotFoundError(f"Image not found: {path}")
        
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        
        return img, q

# ================= TRANSFORMS =================
class RandomGamma:
    def __init__(self, gamma_min=0.7, gamma_max=1.5):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def __call__(self, img):
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return F.adjust_gamma(img, gamma)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(0.6, 0.6, 0.4, 0.06),
    RandomGamma(0.7, 1.6),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================= TRAINING =================
def train():
    download_and_extract_dataset()
    
    # Load CSV and create train/val split
    df = pd.read_csv(CSV_PATH)
    num_samples = len(df)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    split_idx = int(TRAIN_VAL_SPLIT * num_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    log.info(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val")
    
    # Create datasets with indices
    train_ds = PoseDataset(CSV_PATH, IMG_DIR, indices=train_indices, transform=train_transform)
    val_ds = PoseDataset(CSV_PATH, IMG_DIR, indices=val_indices, transform=val_transform)
    
    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=6, pin_memory=True)
    
    model = PoseModel().to(DEVICE)
    criterion = QuaternionCosineLoss() if USE_COSINE_QUAT_LOSS else QuaternionMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val = float("inf")
    patience = 40
    patience_counter = 0
    
    log.info(f"Starting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        
        for i, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                preds = preds / (torch.norm(preds, dim=1, keepdim=True) + 1e-8)
                
                # *** CRITICAL FIX: Quaternion Ambiguity (q vs -q) Alignment ***
                # Flip PREDICTIONS to align with targets, not vice versa
                dot = torch.sum(preds * targets, dim=1, keepdim=True)
                preds = torch.where(dot < 0, -preds, preds)
                # ************************************************************
                
                loss = criterion(preds, targets)
                loss = loss / ACCUM_STEPS
            
            scaler.scale(loss).backward()
            running += loss.item() * ACCUM_STEPS
            
            if (i + 1) % ACCUM_STEPS == 0:
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
                    preds = preds / (torch.norm(preds, dim=1, keepdim=True) + 1e-8)
                    
                    # *** CRITICAL FIX: Quaternion Ambiguity (q vs -q) Alignment ***
                    dot = torch.sum(preds * targets, dim=1, keepdim=True)
                    preds = torch.where(dot < 0, -preds, preds)
                    # ************************************************************
                    
                    vloss += criterion(preds, targets).item()
        
        avg_val = vloss / max(1, len(val_loader))
        scheduler.step()
        
        log.info(f"Epoch {epoch} -> train={avg_train:.4f} val={avg_val:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
        log_gpu_usage()
        
        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "pose_model_best.pt"))
            
            # Export ONNX
            dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
            model.eval()
            onnx_path = os.path.join(MODEL_DIR, "pose_model_best.onnx")
            torch.onnx.export(
                model, dummy_input, onnx_path,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
            )
            log.info(f"ONNX model exported to {onnx_path}")
            log.info("Saved NEW BEST model")
        else:
            patience_counter += 1
            log.info(f"No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                log.info("EARLY STOPPING")
                break
    
    # Save final checkpoint
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "pose_model_final.pt"))
    log.info(f"Training complete. Best val: {best_val}")
    
    # Export final ONNX
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    model.eval()
    onnx_path = os.path.join(MODEL_DIR, "pose_model_final.onnx")
    torch.onnx.export(
        model, dummy_input, onnx_path,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    log.info(f"Final ONNX model exported to {onnx_path}")

# ================= ENTRY POINT =================
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support() 
    train()