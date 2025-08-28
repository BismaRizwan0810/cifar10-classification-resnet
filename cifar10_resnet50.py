# cifar10_resnet50.py
import os, shutil, random, glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from pathlib import Path

# ========== SET DATASET ROOT ==========
KAGGLE_ROOT = r"D:\Cifar10_kaggle\Dataset\cifar-10"
labels_csv = os.path.join(KAGGLE_ROOT, "trainLabels.csv")
train_img_dir = os.path.join(KAGGLE_ROOT, "train")

if not os.path.exists(labels_csv):
    raise FileNotFoundError(f"trainLabels.csv not found at {labels_csv}")
if not os.path.exists(train_img_dir):
    raise FileNotFoundError(f"train folder not found at {train_img_dir}")

print("✅ Using dataset from:", KAGGLE_ROOT)

# output subset banane ke liye
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "dataset")

# ========== PARAMETERS ==========
K = 33                 # total images per class
TEST_PER_CLASS = 8     # test images per class
SEED = 42
random.seed(SEED)

# =======================
# 1) PREPARE SUBSET
# =======================
df = pd.read_csv(labels_csv)
classes = sorted(df['label'].unique())
print("Classes found:", classes)

# make folders
for split in ["train", "test"]:
    for c in classes:
        Path(os.path.join(OUTPUT_ROOT, split, c)).mkdir(parents=True, exist_ok=True)

# copy limited images
for c in classes:
    ids = df[df['label'] == c]['id'].tolist()
    random.shuffle(ids)
    pick = ids[:K]
    test_ids = pick[:TEST_PER_CLASS]
    train_ids = pick[TEST_PER_CLASS:]

    for i in train_ids:
        src = os.path.join(train_img_dir, f"{i}.png")
        dst = os.path.join(OUTPUT_ROOT, "train", c, f"{i}.png")
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

    for i in test_ids:
        src = os.path.join(train_img_dir, f"{i}.png")
        dst = os.path.join(OUTPUT_ROOT, "test", c, f"{i}.png")
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

print("✅ Subset ready at:", os.path.abspath(OUTPUT_ROOT))

# =======================
# 2) TRAIN MODEL
# =======================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(OUTPUT_ROOT, "train"),
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(OUTPUT_ROOT, "train"),
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(OUTPUT_ROOT, "test"),
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
])

# Model
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
base = ResNet50(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
base.trainable = False   # freeze backbone
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n=== Training classification head ===")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    verbose=1
)

# Evaluate
print("\n=== Evaluation on TEST ===")
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"✅ Test Accuracy: {test_acc:.3f}")

# Predict samples
def predict_one(path):
    img = keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    pred = model.predict(arr, verbose=0)[0]
    idx = np.argmax(pred)
    return os.path.basename(path), class_names[idx], float(pred[idx])

print("\nSample Predictions:")
for c in class_names:
    files = glob.glob(os.path.join(OUTPUT_ROOT, "test", c, "*.*"))
    if files:
        fname, label, conf = predict_one(files[0])
        print(f"{fname:>15}  →  {label}  ({conf:.2f})")

# Save model
model.save("resnet50_cifar10_subset.keras")
print("\nModel saved: resnet50_cifar10_subset.keras")
