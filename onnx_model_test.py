# inference_quat_onnx_actual.py
"""
ONNX inference & visualization for quaternion model.
Prints predicted and actual quaternions/Euler angles and angular error.
Optional nearest dataset image display.
"""
import os, glob, re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import onnxruntime as ort
from torchvision import transforms
from scipy.spatial.transform import Rotation as R

# ================= CONFIG =================
TEST_DIR = r"C:\Users\me\Desktop\pose-prediction\dataset\test"
DATASET_DIR = r"C:\Users\me\Desktop\pose-prediction\dataset"
ONNX_MODEL = "model.onnx"
FIND_NEAREST = True
CSV_GLOB = "dataset_csv/rotations_20251129_213001.csv"

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ================= HELPERS =================
def preprocess_image(fname):
    path = os.path.join(TEST_DIR, fname)
    if not os.path.exists(path):
        path = os.path.join(TEST_DIR, os.path.basename(fname)).replace(".png", ".jpg")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    img = Image.open(path).convert("RGBA")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255,255,255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert("RGB")
    tensor = transform(img).unsqueeze(0).numpy().astype(np.float32)
    return tensor, img

def parse_quat_from_filename(fname):
    """Parse quaternion from filename: wX_xX_yX_zX"""
    m = re.search(
        r"w([+-]?\d+(?:\.\d+)?)_?x([+-]?\d+(?:\.\d+)?)_?y([+-]?\d+(?:\.\d+)?)_?z([+-]?\d+(?:\.\d+)?)",
        fname, flags=re.IGNORECASE
    )
    if m:
        return np.array([float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(1))], dtype=float)
    return None

def quat_to_filename_format(q):
    """Convert quaternion [x, y, z, w] to w.._x.._y.._z.. format"""
    return f"w{q[3]:.6f}_x{q[0]:.6f}_y{q[1]:.6f}_z{q[2]:.6f}"

# ================= LOAD DATASET QUATS (optional nearest lookup) =================
dataset_quats = []
dataset_files = []

if FIND_NEAREST:
    import pandas as pd
    csvs = glob.glob(CSV_GLOB)
    if csvs:
        df = pd.read_csv(csvs[0])
        if {'x','y','z','w','filename'}.issubset(df.columns):
            for _, r in df.iterrows():
                q = np.array([r['x'], r['y'], r['z'], r['w']], dtype=float)
                q /= np.linalg.norm(q) + 1e-8
                dataset_quats.append(q)
                candidate = os.path.join(DATASET_DIR, r['filename']).replace(".png", ".jpg")
                if not os.path.exists(candidate):
                    candidate = os.path.join(DATASET_DIR, os.path.basename(r['filename']))
                dataset_files.append(candidate)

    if len(dataset_quats) == 0:
        # fallback: parse filenames
        for f in os.listdir(DATASET_DIR):
            if f.lower().endswith((".png",".jpg",".jpeg")):
                parsed = parse_quat_from_filename(f)
                if parsed is not None:
                    parsed /= np.linalg.norm(parsed) + 1e-8
                    dataset_quats.append(parsed)
                    dataset_files.append(os.path.join(DATASET_DIR, f))

    if len(dataset_quats) == 0:
        print("No dataset quaternions found — disabling nearest-image lookup.")
        FIND_NEAREST = False
    else:
        dataset_quats = np.array(dataset_quats)
        print(f"Loaded {len(dataset_quats)} dataset quaternions for nearest lookup.")

# ================= INITIALIZE ONNX SESSION =================
session = ort.InferenceSession(ONNX_MODEL)
input_name = session.get_inputs()[0].name

# ================= INFER & VISUALIZE =================
test_images = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))])
if not test_images:
    raise RuntimeError("No test images in TEST_DIR")

for fname in test_images:
    inp_tensor, pil_img = preprocess_image(fname)

    # ---- actual quaternion from test filename ----
    actual_q = parse_quat_from_filename(fname)
    if actual_q is not None:
        actual_q /= np.linalg.norm(actual_q) + 1e-8
        actual_euler = R.from_quat(actual_q).as_euler('xyz', degrees=True)
    else:
        actual_euler = None

    # ---- ONNX inference ----
    pred = session.run(None, {input_name: inp_tensor})[0][0]
    pred /= np.linalg.norm(pred) + 1e-8
    pred_euler = R.from_quat(pred).as_euler('xyz', degrees=True)

    # ---- optional nearest dataset image ----
    if FIND_NEAREST and len(dataset_quats) > 0:
        dists = np.linalg.norm(dataset_quats - pred, axis=1)
        idx = int(np.argmin(dists))
        closest_q = dataset_quats[idx]
        closest_file = dataset_files[idx]
        closest_img = Image.open(closest_file).convert("RGB") if os.path.exists(closest_file) else None
        r_rel = R.from_quat(pred) * R.from_quat(closest_q).inv()
        angle_deg = r_rel.magnitude() * (180.0/np.pi)
    else:
        closest_q = None
        closest_img = None
        angle_deg = None

    # ---- PRINT RESULTS ----
    print(f"\n=== {fname} ===")
    if actual_q is not None:
        print("Actual quaternion [x, y, z, w]:", actual_q)
        print("Actual quaternion filename-style:", quat_to_filename_format(actual_q))
        print("Actual Euler angles [deg]:", np.round(actual_euler, 2))
    else:
        print("Actual quaternion not found in filename!")

    print("Predicted quaternion [x, y, z, w]:", pred)
    print("Predicted quaternion filename-style:", quat_to_filename_format(pred))
    print("Predicted Euler angles [deg]:", np.round(pred_euler, 2))

    if closest_q is not None:
        print("Closest dataset quaternion [x, y, z, w]:", closest_q)
        print("Closest dataset quaternion filename-style:", quat_to_filename_format(closest_q))
        print(f"Angular error with closest dataset [deg]: {angle_deg:.2f}")

    # ---- PLOT ----
    plt.figure(figsize=(8,4))
    plt.suptitle(f"{fname}")
    plt.subplot(1,2,1)
    plt.imshow(pil_img)
    plt.title("Test Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    if closest_img is not None:
        plt.imshow(closest_img)
        plt.title("Closest Dataset Image")
    else:
        plt.text(0.5, 0.5, "No nearest-image available", ha='center', va='center', fontsize=12)
    plt.axis("off")
    plt.show()
