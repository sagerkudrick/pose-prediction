# # inference_quat_only_with_angle_loss.py
# """
# Inference & visualization for quaternion model with angular difference.
# Looks up nearest dataset quaternion (actual) by either:
#  - CSV in dataset_csv/rotations_*.csv (preferred), or
#  - parsing filenames like X12.3_Y-45.0_Z90.0.png inside DATASET_DIR
# Displays angle difference between predicted and actual rotation in degrees.
# """
# import os, glob, re
# import numpy as np
# import pandas as pd
# from PIL import Image
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from scipy.spatial.transform import Rotation as R

# # ============== CONFIG ==============
# TEST_DIR = r"C:\Users\me\Desktop\pose-prediction\dataset\test"
# DATASET_DIR = r"C:\Users\me\Desktop\pose-prediction\dataset"
# MODEL_PATH = "pose_model_final.pt"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# FIND_NEAREST = True
# CSV_GLOB = "rotations_20251122_124605.csv"

# # ============== MODEL ==============
# class PoseModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         backbone = models.resnet18(pretrained=True)
#         backbone.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
#         backbone.fc = nn.Sequential(
#             nn.Linear(backbone.fc.in_features, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 4)
#         )
#         self.backbone = backbone

#     def forward(self, x):
#         return self.backbone(x)

# # ============== HELPERS ==============
# def parse_quat_from_filename(fname):
#     m = re.search(
#         r"w([+-]?\d+(?:\.\d+)?)_?x([+-]?\d+(?:\.\d+)?)_?y([+-]?\d+(?:\.\d+)?)_?z([+-]?\d+(?:\.\d+)?)",
#         fname, flags=re.IGNORECASE
#     )
#     if m:
#         return np.array([float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(1))], dtype=float)
#     return None

# def quat_angle_diff(q1, q2):
#     """Compute angular difference between two quaternions in degrees."""
#     q1 = q1 / (np.linalg.norm(q1) + 1e-8)
#     q2 = q2 / (np.linalg.norm(q2) + 1e-8)
#     dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
#     angle_rad = 2 * np.arccos(dot)
#     return np.degrees(angle_rad)

# # ============== LOAD MODEL ==============
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3)
# ])

# model = PoseModel().to(DEVICE)
# state = torch.load(MODEL_PATH, map_location=DEVICE)
# model.load_state_dict(state)
# model.eval()
# print("Loaded model:", MODEL_PATH, "on", DEVICE)

# # ============== LOAD DATASET QUATS ==============
# dataset_quats = []
# dataset_files = []

# if FIND_NEAREST:
#     csvs = glob.glob(CSV_GLOB)
#     if csvs:
#         df = pd.read_csv(csvs[0])
#         if {'x','y','z','w','filename'}.issubset(df.columns):
#             for _, r in df.iterrows():
#                 q = np.array([r['x'], r['y'], r['z'], r['w']], dtype=float)
#                 q /= np.linalg.norm(q) + 1e-8
#                 dataset_quats.append(q)
#                 candidate = os.path.join(DATASET_DIR, r['filename']).replace(".png",".jpg")
#                 if not os.path.exists(candidate):
#                     candidate = os.path.join(DATASET_DIR, os.path.basename(r['filename']))
#                 dataset_files.append(candidate)

#     if len(dataset_quats) == 0:
#         for f in os.listdir(DATASET_DIR):
#             if f.lower().endswith((".png",".jpg",".jpeg")):
#                 parsed = parse_quat_from_filename(f)
#                 if parsed is not None:
#                     parsed /= np.linalg.norm(parsed) + 1e-8
#                     dataset_quats.append(parsed)
#                     dataset_files.append(os.path.join(DATASET_DIR, f))

#     if len(dataset_quats) == 0:
#         print("No dataset quaternions found — disabling nearest-image lookup.")
#         FIND_NEAREST = False
#     else:
#         dataset_quats = np.array(dataset_quats)
#         print(f"Loaded {len(dataset_quats)} dataset quaternions for nearest lookup.")

# # ============== INFER & SHOW ==============
# test_images = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))])
# if not test_images:
#     raise RuntimeError("No test images in TEST_DIR")

# for fname in test_images:
#     test_path = os.path.join(TEST_DIR, fname)
#     img = Image.open(test_path).convert("RGB")
#     inp = transform(img).unsqueeze(0).to(DEVICE)
    
#     with torch.no_grad():
#         pred = model(inp)[0].cpu().numpy()
#         pred /= np.linalg.norm(pred) + 1e-8

#     # nearest dataset quaternion (optional)
#     if FIND_NEAREST:
#         dists = np.linalg.norm(dataset_quats - pred, axis=1)
#         idx = int(np.argmin(dists))
#         closest_file = dataset_files[idx] if os.path.exists(dataset_files[idx]) else None
#         closest_img = Image.open(closest_file).convert("RGB") if closest_file else None
#         closest_q = dataset_quats[idx] if closest_file else None
#     else:
#         closest_img = None
#         closest_q = None

#     # convert quaternions to Euler angles
#     pred_euler = R.from_quat(pred).as_euler('xyz', degrees=True)
#     if closest_q is not None:
#         closest_euler = R.from_quat(closest_q).as_euler('xyz', degrees=True)
#         angle_loss = quat_angle_diff(pred, closest_q)

#     # print results
#     print(f"\n=== {fname} ===")
#     print("Predicted quaternion [x, y, z, w]:", pred)
#     print("Predicted Euler angles [deg]:", pred_euler)
#     if closest_q is not None:
#         print("Closest dataset quaternion [x, y, z, w]:", closest_q)
#         print("Closest dataset Euler angles [deg]:", closest_euler)
#         print("Angular difference [deg]:", angle_loss)

#     # plot
#     plt.figure(figsize=(8,4))
#     plt.suptitle(f"{fname} | Angle diff: {angle_loss:.2f}°" if closest_q is not None else fname)
    
#     plt.subplot(1,2,1)
#     plt.imshow(img)
#     plt.title("Test")
#     plt.axis("off")

#     plt.subplot(1,2,2)
#     if closest_img is not None:
#         plt.imshow(closest_img)
#         plt.title("Closest dataset image")
#     else:
#         plt.text(0.5, 0.5, "No nearest-image available", ha='center', va='center', fontsize=12)
#     plt.axis("off")
#     plt.show()

# inference_quat_single_image.py
"""
Inference & visualization for quaternion model on a single image.
Displays predicted quaternion and Euler angles, optionally
compares with nearest dataset quaternion for angular difference.
"""
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy.spatial.transform import Rotation as R
import pandas as pd

# ============== CONFIG ==============
DATASET_DIR = r"C:\Users\me\Desktop\pose-prediction\dataset"
MODEL_PATH = "pose_model_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIND_NEAREST = True
CSV_GLOB = "rotations_20251203_150653.csv"

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
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# ============== HELPERS ==============
def quat_angle_diff(q1, q2):
    """Compute angular difference between two quaternions in degrees."""
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    q2 = q2 / (np.linalg.norm(q2) + 1e-8)
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    angle_rad = 2 * np.arccos(dot)
    return np.degrees(angle_rad)

# ============== LOAD MODEL ==============
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

model = PoseModel().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print("Loaded model:", MODEL_PATH, "on", DEVICE)

# ============== LOAD DATASET QUATS ==============
dataset_quats = []
dataset_files = []

if FIND_NEAREST:
    csvs = glob.glob(CSV_GLOB)
    if csvs:
        df = pd.read_csv(csvs[0])
        if {'x','y','z','w','filename'}.issubset(df.columns):
            for _, r in df.iterrows():
                q = np.array([r['x'], r['y'], r['z'], r['w']], dtype=float)
                q /= np.linalg.norm(q) + 1e-8
                dataset_quats.append(q)
                candidate = os.path.join(DATASET_DIR, r['filename']).replace(".png",".jpg")
                if not os.path.exists(candidate):
                    candidate = os.path.join(DATASET_DIR, os.path.basename(r['filename']))
                dataset_files.append(candidate)

    if len(dataset_quats) == 0:
        print("No dataset quaternions found — disabling nearest-image lookup.")
        FIND_NEAREST = False
    else:
        dataset_quats = np.array(dataset_quats)
        print(f"Loaded {len(dataset_quats)} dataset quaternions for nearest lookup.")

# ============== INFER ON SINGLE IMAGE ==============
def infer_image(image_path):
    img = Image.open(image_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred = model(inp)[0].cpu().numpy()
        pred /= np.linalg.norm(pred) + 1e-8

    # nearest dataset quaternion
    closest_q = None
    closest_img = None
    angle_loss = None
    if FIND_NEAREST:
        dists = np.linalg.norm(dataset_quats - pred, axis=1)
        idx = int(np.argmin(dists))
        closest_file = dataset_files[idx] if os.path.exists(dataset_files[idx]) else None
        if closest_file:
            closest_img = Image.open(closest_file).convert("RGB")
            closest_q = dataset_quats[idx]
            angle_loss = quat_angle_diff(pred, closest_q)

    # convert to Euler
    pred_euler = R.from_quat(pred).as_euler('xyz', degrees=True)
    closest_euler = R.from_quat(closest_q).as_euler('xyz', degrees=True) if closest_q is not None else None

    # print results
    print(f"\n=== {os.path.basename(image_path)} ===")
    print("Predicted quaternion [x, y, z, w]:", pred)
    print("Predicted Euler angles [deg]:", pred_euler)
    if closest_q is not None:
        print("Closest dataset quaternion [x, y, z, w]:", closest_q)
        print("Closest dataset Euler angles [deg]:", closest_euler)
        print("Angular difference [deg]:", angle_loss)

    # plot
    plt.figure(figsize=(8,4))
    plt.suptitle(f"{os.path.basename(image_path)} | Angle diff: {angle_loss:.2f}°" if angle_loss is not None else os.path.basename(image_path))
    
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Input image")
    plt.axis("off")

    plt.subplot(1,2,2)
    if closest_img is not None:
        plt.imshow(closest_img)
        plt.title("Closest dataset image")
    else:
        plt.text(0.5, 0.5, "No nearest-image available", ha='center', va='center', fontsize=12)
    plt.axis("off")
    plt.show()


# ================= USAGE =================
if __name__ == "__main__":
    test_image = r"C:\Users\me\Desktop\pose-prediction\dataset\test\download.png"  # <-- change this
    infer_image(test_image)

