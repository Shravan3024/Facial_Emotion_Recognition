import cv2
import numpy as np
import os
from tensorflow.keras.applications import efficientnet, mobilenet_v2

# These globals will be set from app.py
MODEL_INPUT_SHAPE = None     # (H, W, C)
MODEL_PREPROCESS_FN = None   # EfficientNet / MobileNet / None
MODEL_COLOR_MODE = "rgb"     # "rgb" or "grayscale"


def configure_preprocessing(input_shape, model_filename):
    """
    Called once from app.py to configure the correct preprocessing
    depending on model type.
    """
    global MODEL_INPUT_SHAPE, MODEL_PREPROCESS_FN, MODEL_COLOR_MODE

    MODEL_INPUT_SHAPE = input_shape
    _, H, W, C = input_shape

    # Determine color mode
    MODEL_COLOR_MODE = "rgb" if C == 3 else "grayscale"

    lower = model_filename.lower()
    
    # Determine preprocessing
    if "efficient" in lower:
        MODEL_PREPROCESS_FN = efficientnet.preprocess_input
    elif "mobilenet" in lower:
        MODEL_PREPROCESS_FN = mobilenet_v2.preprocess_input
    else:
        MODEL_PREPROCESS_FN = None    # simple /255 normalization


def preprocess_image_for_model(face_bgr):
    """
    Final preprocessing function used by app.py.
    Automatically adapts to:
        - RGB / grayscale
        - EfficientNet / MobileNet / simple model
        - Target input size from model
        - Histogram Equalization
        - Margins
    """

    global MODEL_INPUT_SHAPE, MODEL_COLOR_MODE, MODEL_PREPROCESS_FN

    if MODEL_INPUT_SHAPE is None:
        raise ValueError("❌ Preprocessing not configured! Call configure_preprocessing() first.")

    _, H, W, C = MODEL_INPUT_SHAPE

    # ----- 1. Add face margin -----
    h, w = face_bgr.shape[:2]
    margin = int(0.18 * h)

    y1 = max(0, -margin)
    x1 = max(0, -margin)
    y2 = min(h + margin, h)
    x2 = min(w + margin, w)

    face_bgr = face_bgr[y1:y2, x1:x2]

    # ----- 2. Prepare according to color mode -----
    if MODEL_COLOR_MODE == "grayscale":
        # Convert to grayscale
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        # Histogram equalization
        gray = cv2.equalizeHist(gray)

        # Resize
        gray = cv2.resize(gray, (W, H))

        # Normalize
        arr = gray.astype("float32") / 255.0

        # Add channel dim
        arr = np.expand_dims(arr, axis=-1)

    else:
        # RGB
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # Histogram Equalization in LAB
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        L = cv2.equalizeHist(L)
        lab = cv2.merge((L, A, B))
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Resize
        rgb = cv2.resize(rgb, (W, H))

        arr = rgb.astype("float32")

        # Apply special model preprocess
        if MODEL_PREPROCESS_FN is None:
            arr = arr / 255.0
        else:
            arr = MODEL_PREPROCESS_FN(arr)

    # ----- 3. Add batch dimension -----
    arr = np.expand_dims(arr, axis=0)

    return arr
