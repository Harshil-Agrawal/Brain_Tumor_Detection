import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model("BrainTumor10epochs.h5")

# Load and preprocess the image
image = cv2.imread("pred\pred4.jpg")
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)

# Ensure the input has the correct shape and add a batch dimension
input_image = np.expand_dims(img, axis=0)

# Make predictions
predictions = model.predict(input_image)
#print(predictions)
if int(predictions[0][0]) == 1:
    print("Tumor Detected")
else:
    print("Tumor not Detected")    