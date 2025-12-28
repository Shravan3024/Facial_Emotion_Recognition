import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import cv2
import av

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import efficientnet, mobilenet_v2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# -------------------------------------------------------
# FIND ANY VALID MODEL FILE
# -------------------------------------------------------
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
    full = os.path.join(MODEL_DIR, m)
    if os.path.exists(full):
        model_path = full
        break

if model_path is None:
    # fallback: pick first .h5
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".h5"):
            model_path = os.path.join(MODEL_DIR, f)
            break

if model_path is None:
    st.error("No .h5 model found in /model folder.")
    st.stop()

st.sidebar.write(f"**Loaded Model:** `{os.path.basename(model_path)}`")

model = load_model(model_path)
input_shape = model.input_shape   # (None,H,W,C)
_, H, W, C = input_shape

# -------------------------------------------------------
# DETERMINE PREPROCESSING
# -------------------------------------------------------
lower = os.path.basename(model_path).lower()

if "efficient" in lower:
    preprocess_fn = efficientnet.preprocess_input
elif "mobilenet" in lower:
    preprocess_fn = mobilenet_v2.preprocess_input
else:
    preprocess_fn = None        # simple normalization

color_mode = "rgb" if C == 3 else "grayscale"
target_size = (H, W)

# -------------------------------------------------------
# LOAD LABELS
# -------------------------------------------------------
class_json = os.path.join(MODEL_DIR, "class_indices.json")
if os.path.exists(class_json):
    import json
    with open(class_json, "r") as f:
        idxs = json.load(f)
    inv = {int(v): k for k, v in idxs.items()}
    LABELS = [inv[i] for i in sorted(inv.keys())]
else:
    LABELS = ['angry','disgust','fear','happy','neutral','sad','surprise']

# -------------------------------------------------------
# FACE DETECTOR
# -------------------------------------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------------------------------------
# FACE PREPROCESSING (IDENTICAL TO realtime_app)
# -------------------------------------------------------
def preprocess_face(face_bgr):

    if color_mode == "grayscale":
        g = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(g, target_size)
        arr = resized.astype("float32") / 255.0
        arr = np.expand_dims(arr, -1)

    else:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, target_size)
        arr = resized.astype("float32")

        if preprocess_fn is None:
            arr = arr / 255.0
        else:
            arr = preprocess_fn(arr)

    return np.expand_dims(arr, 0)  # (1,H,W,C)



# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")
st.title("😊 Facial Emotion Recognition App")

mode = st.sidebar.selectbox("Choose Mode:", ["Image Upload", "Webcam Detection"])



# ======================================================
# IMAGE UPLOAD MODE
# ======================================================
if mode == "Image Upload":
    st.header("📷 Upload an Image")

    uploaded = st.file_uploader("Upload face image", type=["jpg", "png", "jpeg"])

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.2, 4, minSize=(50, 50))

        if len(faces) == 0:
            st.warning("No face detected.")
            st.image(img, channels="BGR")
        else:
            x, y, w, h = faces[0]

            # Expand slightly
            pad = int(0.12 * w)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad)
            y2 = min(img.shape[0], y + h + pad)

            face = img[y1:y2, x1:x2]

            inp = preprocess_face(face)
            preds = model.predict(inp, verbose=0)[0]
            label = LABELS[int(np.argmax(preds))]
            conf = np.max(preds)

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            st.image(img, channels="BGR")
            st.success(f"Prediction: **{label}**  ({conf:.2f})")



# ======================================================
# REALTIME WEBCAM MODE
# ======================================================
if mode == "Webcam Detection":
    st.header("🎥 Live Emotion Detection")

    class EmotionTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                pad = int(0.15 * w)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(img.shape[1], x + w + pad)
                y2 = min(img.shape[0], y + h + pad)

                face = img[y1:y2, x1:x2]

                try:
                    inp = preprocess_face(face)
                    preds = model.predict(inp, verbose=0)[0]
                    idx = int(np.argmax(preds))
                    conf = np.max(preds)
                    label = LABELS[idx]
                except:
                    label, conf = "err", 0.0

                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            return img

    webrtc_streamer(
        key="emotion-detect",
        video_transformer_factory=EmotionTransformer,
        media_stream_constraints={"video": True, "audio": False}
    )
