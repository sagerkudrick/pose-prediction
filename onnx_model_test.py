# test_onnx_random.py
import onnxruntime as ort
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import os
import random

# Paths
ONNX_PATH = "resnet18_web.onnx"
CSV_PATH = "dataset_csv/rotations_20251122_124605.csv"
IMG_DIR = "dataset"

# Transforms (same as validation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load CSV
df = pd.read_csv(CSV_PATH)

# Pick 5 random rows
df_sample = df.sample(n=5, random_state=1221)

# Initialize ONNX Runtime session
session = ort.InferenceSession(ONNX_PATH)

# Helper to load and preprocess images
def preprocess_image(filename):
    path = os.path.join(IMG_DIR, filename)
    if not os.path.exists(path):
        path = os.path.join(IMG_DIR, os.path.basename(filename)).replace(".png", ".jpg")
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    img = Image.open(path).convert("RGBA")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0,0,0))  # black instead of white
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert("RGB")


    img = val_transform(img)
    return img.unsqueeze(0).numpy().astype(np.float32)  # add batch dim

# Inference
preds = []
for idx, row in df_sample.iterrows():
    img_tensor = preprocess_image(row['filename'])
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: img_tensor})[0]
    pred_q = out[0]
    pred_q = pred_q / (np.linalg.norm(pred_q) + 1e-8)
    preds.append(pred_q)

preds = np.array(preds)
print("Predictions shape:", preds.shape)
print("Predicted quaternions:\n", preds)

# Optional: compare with ground truth
gt_quats = df_sample[['x','y','z','w']].to_numpy()
diff = np.abs(preds - gt_quats)
print("Mean absolute difference per component:", diff.mean(axis=0))
