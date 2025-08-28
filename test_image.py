import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image

# ==== LOAD MODEL ====
model = keras.models.load_model("resnet50_cifar10_subset.keras")

# ==== CIFAR-10 CLASS NAMES ====
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ==== TEST IMAGE PATH ====
IMG_PATH = r"D:\Cifar10_kaggle\Dataset\test\deer\1204.png"  # apni test image ka path yahan likho
IMG_SIZE = (224, 224)

# ==== LOAD & PREPROCESS IMAGE ====
img = Image.open(IMG_PATH).convert("RGB")
img = img.resize(IMG_SIZE)
img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)  # shape (1,224,224,3)
img_array = preprocess_input(img_array)

# ==== PREDICT ====
pred = model.predict(img_array)
class_id = np.argmax(pred)
confidence = np.max(pred)

print(f"Predicted: {class_names[class_id]}  (confidence: {confidence:.2f})")
