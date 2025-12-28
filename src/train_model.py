# src/train_model.py
import os, json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import build_fer_cnn

# compute project root (parent of src)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# paths
train_dir = os.path.join(BASE_DIR, "dataset", "train")
val_dir   = os.path.join(BASE_DIR, "dataset", "val")
test_dir  = os.path.join(BASE_DIR, "dataset", "test")
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

# hyperparams
input_shape = (48,48,1)
num_classes = 7
batch_size = 64
epochs = 50
lr = 1e-3

# check whether val_dir exists; if not we'll use validation_split on train
use_val_dir = os.path.isdir(val_dir)

if use_val_dir:
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=10,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       zoom_range=0.1,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(48,48), color_mode='grayscale',
        batch_size=batch_size, class_mode='categorical'
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=(48,48), color_mode='grayscale',
        batch_size=batch_size, class_mode='categorical', shuffle=False
    )
else:
    # use validation_split
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=10,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       zoom_range=0.1,
                                       horizontal_flip=True,
                                       fill_mode='nearest',
                                       validation_split=0.1)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(48,48), color_mode='grayscale',
        batch_size=batch_size, class_mode='categorical', subset='training'
    )
    val_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(48,48), color_mode='grayscale',
        batch_size=batch_size, class_mode='categorical', subset='validation', shuffle=False
    )

# save class indices
class_indices_path = os.path.join(model_dir, "class_indices.json")
with open(class_indices_path, "w") as f:
    json.dump(train_gen.class_indices, f)
print("Saved class indices:", train_gen.class_indices)

# build & compile model
model = build_fer_cnn(input_shape=input_shape, num_classes=num_classes)
model.compile(optimizer=Adam(learning_rate=lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# callbacks
checkpoint_path = os.path.join(model_dir, "fer_model.h5")
callbacks = [
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

final_model_path = os.path.join(model_dir, "fer_model_final.h5")
model.save(final_model_path)
print("Saved final model to:", final_model_path)
