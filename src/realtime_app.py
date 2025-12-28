# src/realtime_app.py
import os
import json
import time
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import efficientnet, mobilenet_v2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ---------- auto-pick model ----------
possible_models = [
    "fer_efficientnet_final.h5",
    "fer_efficientnet.h5",
    "fer_mobilenet_final.h5",
    "fer_mobilenet.h5",
    "fer_model_final.h5",
    "fer_model.h5"
]

model_path = None
for m in possible_models:
    p = os.path.join(MODEL_DIR, m)
    if os.path.exists(p):
        model_path = p
        break

# fallback: first .h5 in model folder
if model_path is None:
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".h5"):
            model_path = os.path.join(MODEL_DIR, f)
            break

if model_path is None:
    raise FileNotFoundError("No model (.h5) found in model/ folder. Train a model first.")

print("Loading model:", model_path)
model = load_model(model_path)
input_shape = model.input_shape  # (None, H, W, C)
print("Model input shape:", input_shape)

# ---------- infer preprocessing from model name ----------
lower_name = os.path.basename(model_path).lower()
if "efficient" in lower_name:
    preprocess_fn = efficientnet.preprocess_input
    print("Using EfficientNet preprocess")
elif "mobilenet" in lower_name:
    preprocess_fn = mobilenet_v2.preprocess_input
    print("Using MobileNetV2 preprocess")
else:
    preprocess_fn = None
    print("Using simple rescale (1./255) preprocess")

# ---------- infer target size & color mode ----------
try:
    _, H, W, C = input_shape
    target_size = (int(H), int(W))
    color_mode = 'rgb' if int(C) == 3 else 'grayscale'
except Exception:
    target_size = (48, 48)
    color_mode = 'grayscale'

print("Target size:", target_size, "Color mode:", color_mode)

# ---------- load labels ----------
class_json = os.path.join(MODEL_DIR, "class_indices.json")
if os.path.exists(class_json):
    with open(class_json, "r") as f:
        class_indices = json.load(f)
    inv = {int(v): k for k, v in class_indices.items()}
    labels = [inv[i] for i in sorted(inv.keys())]
else:
    # fallback: default order
    labels = ['angry','disgust','fear','happy','neutral','sad','surprise']
print("Labels:", labels)

# ---------- face detector ----------
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("Warning: Haar cascade not found or failed to load.")

# ---------- preprocess helper ----------
def preprocess_face_for_model(face_bgr):
    """
    Input: face in BGR (OpenCV)
    Returns: array shaped (1,H,W,C) ready for model.predict
    """
    # if grayscale model requested, convert accordingly
    if color_mode == 'grayscale':
        face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(face_gray, target_size)
        arr = resized.astype('float32')
        arr = arr / 255.0
        arr = np.expand_dims(arr, axis=-1)  # H,W,1
    else:
        # color model expects rgb
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(face_rgb, target_size)
        arr = resized.astype('float32')
        if preprocess_fn is None:
            arr = arr / 255.0
        else:
            # preprocess_input expects batch axis usually; but works on single image too
            arr = preprocess_fn(arr)
    # add batch dim
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------- webcam ----------
cap_index = 0  # change to 1,2... if your camera uses different index
cap = cv2.VideoCapture(cap_index)
if not cap.isOpened():
    raise RuntimeError(f"Could not open webcam at index {cap_index}. Try changing cap_index.")

print("Press 'q' to quit. Press 's' to save a cropped face image.")

# smoothing for stable label display (optional)
recent_preds = []
SMOOTH_LEN = 5

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue

    # detect faces -- convert to gray for detection (faster)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))

    # sort faces by size (largest first)
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)

    for (x,y,w,h) in faces:
        # expand box slightly for better context
        pad = int(0.1 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face_bgr = frame[y1:y2, x1:x2]

        try:
            inp = preprocess_face_for_model(face_bgr)
            preds = model.predict(inp, verbose=0)[0]
            idx = int(np.argmax(preds))
            conf = float(np.max(preds))
            label = labels[idx] if idx < len(labels) else str(idx)

            # smoothing
            recent_preds.append((label, conf))
            if len(recent_preds) > SMOOTH_LEN:
                recent_preds.pop(0)
            # majority vote / average conf
            top_label = max(set([p[0] for p in recent_preds]), key=lambda l: sum(p[1] for p in recent_preds if p[0]==l))
            avg_conf = np.mean([p[1] for p in recent_preds if p[0] == top_label])
            disp_label = f"{top_label} {avg_conf:.2f}"
        except Exception as e:
            disp_label = "err"
            print("Prediction error:", e)

        # draw rectangle and label
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, disp_label, (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("FER - realtime (q to quit)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('s') and len(faces) > 0:
        # save cropped face for debugging
        (x,y,w,h) = faces[0]
        pad = int(0.1 * w)
        x1 = max(0, x - pad); y1 = max(0, y - pad); x2 = min(frame.shape[1], x + w + pad); y2 = min(frame.shape[0], y + h + pad)
        cropped = frame[y1:y2, x1:x2]
        save_p = os.path.join(BASE_DIR, "model", f"saved_face_{int(time.time())}.jpg")
        cv2.imwrite(save_p, cropped)
        print("Saved face to:", save_p)

cap.release()
cv2.destroyAllWindows()
