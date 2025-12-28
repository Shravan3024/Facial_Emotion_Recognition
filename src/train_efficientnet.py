# src/train_efficientnet.py
"""
EfficientNetB0 transfer-learning for FER dataset (folder layout).
- Expects project root:
  dataset/
    train/
    val/    (optional)
    test/
  model/
- Loads images as RGB and resizes to IMG_SIZE.
- Stage 1: train head with backbone frozen
- Stage 2: unfreeze last N layers and fine-tune
"""

import os, json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, efficientnet
from tensorflow.keras import layers, models, optimizers, callbacks
import tensorflow as tf

# -----------------------
# CONFIG (edit as needed)
# -----------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR   = os.path.join(BASE_DIR, "dataset", "val")   # optional
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

BACKBONE = "EfficientNetB0"   # options: EfficientNetB0, EfficientNetB1, ...
IMG_SIZE = 224                # EfficientNetB0 recommended 224
BATCH = 32
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 15
LEARNING_RATE_STAGE1 = 1e-4
LEARNING_RATE_STAGE2 = 1e-5
FINE_TUNE_UNFREEZE_LAST_N = 60   # how many layers from backbone to unfreeze
NUM_CLASSES = 7
PATIENCE = 5

# Optionally enable mixed precision on compatible hardware (GPU)
# Uncomment if you have suitable GPU and TF configured for mixed precision
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')

# -----------------------
# Data generators
# -----------------------
# We will use the EfficientNet preprocessing function
preprocess_input = efficientnet.preprocess_input

# stronger augmentations
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.12,
    zoom_range=0.12,
    brightness_range=(0.6, 1.4),
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1  # used only if VAL_DIR not present
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

if os.path.isdir(VAL_DIR):
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb',
        batch_size=BATCH, class_mode='categorical', shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb',
        batch_size=BATCH, class_mode='categorical', shuffle=False
    )
else:
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb',
        batch_size=BATCH, class_mode='categorical', subset='training', shuffle=True
    )
    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb',
        batch_size=BATCH, class_mode='categorical', subset='validation', shuffle=False
    )

# Save class indices mapping
class_indices_path = os.path.join(MODEL_DIR, "class_indices.json")
with open(class_indices_path, "w") as f:
    json.dump(train_gen.class_indices, f)
print("Saved class indices:", train_gen.class_indices)

# compute class weights
labels = train_gen.classes
classes_unique = np.unique(labels)
cw = compute_class_weight('balanced', classes=classes_unique, y=labels)
class_weights = {int(classes_unique[i]): float(cw[i]) for i in range(len(classes_unique))}
print("Class weights:", class_weights)

# -----------------------
# Build model
# -----------------------
# Choose backbone
if BACKBONE == "EfficientNetB0":
    backbone = EfficientNetB0(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet', pooling='avg')
else:
    raise ValueError("Only EfficientNetB0 implemented in this script. Change code to add others.")

backbone.trainable = False  # freeze for stage 1

x = backbone.output
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs=backbone.input, outputs=out)
model.compile(optimizer=optimizers.Adam(LEARNING_RATE_STAGE1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Callbacks
checkpoint_path = os.path.join(MODEL_DIR, "fer_efficientnet.h5")
checkpoint = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
es = callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
tensorboard_cb = callbacks.TensorBoard(log_dir=os.path.join(BASE_DIR,"logs","efficientnet"))

# -----------------------
# Stage 1: train head
# -----------------------
print("Stage 1: training head (backbone frozen)")
history1 = model.fit(
    train_gen,
    epochs=EPOCHS_STAGE1,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr, es, tensorboard_cb],
    verbose=1
)

# -----------------------
# Stage 2: fine-tune
# -----------------------
print("Stage 2: fine-tuning last layers of backbone")
backbone.trainable = True
# Freeze all layers except last N
if FINE_TUNE_UNFREEZE_LAST_N is not None:
    total_layers = len(backbone.layers)
    fine_tune_at = max(0, total_layers - int(FINE_TUNE_UNFREEZE_LAST_N))
    for i, layer in enumerate(backbone.layers):
        layer.trainable = (i >= fine_tune_at)

# recompile with smaller LR
model.compile(optimizer=optimizers.Adam(LEARNING_RATE_STAGE2),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history2 = model.fit(
    train_gen,
    epochs=EPOCHS_STAGE2,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr, es, tensorboard_cb],
    verbose=1
)

# Save final model
final_path = os.path.join(MODEL_DIR, "fer_efficientnet_final.h5")
model.save(final_path)
print("Saved final model to:", final_path)
