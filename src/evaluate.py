# src/evaluate.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Attempt to import preprocessing functions for common backbones
from tensorflow.keras.applications import efficientnet, mobilenet_v2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# -------------- pick model file automatically ----------------
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

# fallback: pick any .h5 in model dir if none of above
if model_path is None:
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".h5"):
            model_path = os.path.join(MODEL_DIR, f)
            break

if model_path is None:
    raise FileNotFoundError("No .h5 model file found in model/ folder. Train a model first.")

print("Using model:", model_path)

# ----------------- load model -----------------------
model = load_model(model_path)
input_shape = model.input_shape  # e.g. (None, 224, 224, 3) or (None,48,48,1)
print("Model input shape:", input_shape)

# infer target_size and color_mode from model input_shape
try:
    _, H, W, C = input_shape
    target_size = (int(H), int(W))
    color_mode = 'rgb' if int(C) == 3 else 'grayscale'
except Exception:
    # fallback defaults
    target_size = (48, 48)
    color_mode = 'grayscale'

print("Inferred target_size:", target_size, "color_mode:", color_mode)

# ----------------- choose preprocessing -------------------
# default: simple rescale
def identity_preprocess(x):
    return x

preprocess_fn = None

lower_name = os.path.basename(model_path).lower()
if "efficient" in lower_name:
    print("Using EfficientNet preprocess_input")
    preprocess_fn = efficientnet.preprocess_input
elif "mobilenet" in lower_name:
    print("Using MobileNetV2 preprocess_input")
    preprocess_fn = mobilenet_v2.preprocess_input
else:
    # If filename didn't hint, pick based on input size:
    if target_size[0] >= 160:
        # assume pretrained backbone => use EfficientNet preprocess
        try:
            preprocess_fn = efficientnet.preprocess_input
            print("Assuming EfficientNet-style preprocessing (large input size).")
        except Exception:
            preprocess_fn = identity_preprocess
    else:
        preprocess_fn = identity_preprocess
        print("Using simple rescale (1./255) preprocessing.")

# wrapper generator uses preprocessing_function or rescale
if preprocess_fn == identity_preprocess:
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
else:
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

# ----------------- create generator --------------------
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=target_size,
    color_mode=color_mode,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ----------------- evaluate ---------------------------
print("\nEvaluating model on test set...")
loss, acc = model.evaluate(test_gen, verbose=1)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")

# ----------------- predictions & report ----------------
print("\nGenerating predictions (this may take a bit)...")
y_pred = model.predict(test_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

# labels (index -> label)
class_json = os.path.join(MODEL_DIR, "class_indices.json")
if os.path.exists(class_json):
    with open(class_json, "r") as f:
        class_indices = json.load(f)
    # invert mapping class_name -> index to index -> name
    inv = {int(v): k for k, v in class_indices.items()}
    labels = [inv[i] for i in sorted(inv.keys())]
else:
    labels = list(test_gen.class_indices.keys())

print("\nLabels (index->label):", labels)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=labels, digits=4))

# ----------------- confusion matrix --------------------
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(9,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(cm_path, dpi=150)
plt.close()
print("Saved confusion matrix to:", cm_path)

# ----------------- save CSV with predictions ----------------
filenames = test_gen.filenames
probs = y_pred.tolist()
pred_labels = [labels[i] if i < len(labels) else str(i) for i in y_pred_classes]

df = pd.DataFrame({
    "filename": filenames,
    "true_label_index": y_true,
    "pred_label_index": y_pred_classes,
    "pred_label": pred_labels,
    "probabilities": probs
})
csv_path = os.path.join(MODEL_DIR, "predictions.csv")
df.to_csv(csv_path, index=False)
print("Saved predictions CSV to:", csv_path)

print("\nEvaluation complete.")
