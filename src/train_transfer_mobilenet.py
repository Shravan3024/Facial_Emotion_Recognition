# src/train_transfer_mobilenet.py
import os, json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers, callbacks

# BASE paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
train_dir = os.path.join(BASE_DIR, "dataset", "train")
val_dir   = os.path.join(BASE_DIR, "dataset", "val")
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

# config
IMG_SIZE = 128
BATCH = 32
EPOCHS = 30
LR = 1e-4
NUM_CLASSES = 7

# augmentation - stronger
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.12,
    zoom_range=0.12,
    brightness_range=(0.6,1.4),
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1  # used only if no val_dir exists
)

val_datagen = ImageDataGenerator(rescale=1./255)

# if val folder exists, use it; else use validation_split
if os.path.isdir(val_dir):
    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(IMG_SIZE,IMG_SIZE), color_mode='rgb',
        batch_size=BATCH, class_mode='categorical', shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=(IMG_SIZE,IMG_SIZE), color_mode='rgb',
        batch_size=BATCH, class_mode='categorical', shuffle=False
    )
else:
    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(IMG_SIZE,IMG_SIZE), color_mode='rgb',
        batch_size=BATCH, class_mode='categorical', subset='training'
    )
    val_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(IMG_SIZE,IMG_SIZE), color_mode='rgb',
        batch_size=BATCH, class_mode='categorical', subset='validation', shuffle=False
    )

# compute class weights (robust to imbalance)
labels = train_gen.classes  # integers
class_names = list(train_gen.class_indices.keys())
class_indices = train_gen.class_indices
inv_class_indices = {v:k for k,v in class_indices.items()}

classes_unique = np.unique(labels)
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = {int(classes_unique[i]): float(class_weights[i]) for i in range(len(classes_unique))}
print("Class indices:", class_indices)
print("Class weights:", class_weights_dict)

# save class indices
with open(os.path.join(model_dir, "class_indices.json"), "w") as f:
    json.dump(class_indices, f)

# build model (MobileNetV2 backbone)
backbone = MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3), include_top=False, weights='imagenet', pooling='avg')
backbone.trainable = False  # freeze initially

inp = backbone.input
x = backbone.output
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs=inp, outputs=out)
model.compile(optimizer=optimizers.Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# callbacks
checkpoint = callbacks.ModelCheckpoint(os.path.join(model_dir, "fer_mobilenet.h5"),
                                       monitor='val_accuracy', save_best_only=True, mode='max')
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
es = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# train (stage 1)
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weights_dict,
    callbacks=[checkpoint, reduce_lr, es]
)

# stage 2: fine-tune some backbone layers
backbone.trainable = True
# unfreeze last N layers (small fine-tune)
fine_tune_at = len(backbone.layers) - 40  # unfreeze last 40 layers
for layer in backbone.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(LR/10), loss='categorical_crossentropy', metrics=['accuracy'])
history_ft = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    class_weight=class_weights_dict,
    callbacks=[checkpoint, reduce_lr, es]
)

# save final model
model.save(os.path.join(model_dir, "fer_mobilenet_final.h5"))
print("Saved final model to model/fer_mobilenet_final.h5")
