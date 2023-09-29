# Import Required Modules
import cv2
import os 
import pathlib
from PIL import Image
import numpy as np
#import tensorflow as tf 
from tensorflow import keras

#Listing the paths of the image according to class labels
image_directory = 'datasets/'
data_dir = pathlib.Path(image_directory)
no_tumor = list(data_dir.glob('no/*.jpg'))
yes_tumor = list(data_dir.glob('yes/*.jpg'))

#Images
dataset=[]

#Labels 0 : No (Tumor not Detected)  1 : Yes (Tumor Detected)
label=[]

# Reading the images using opencv and converting it into numpy arrays
for i,image in enumerate(no_tumor):
    image = cv2.imread(str(image))
    image = Image.fromarray(image,'RGB')
    image = image.resize((64,64))
    dataset.append(np.array(image))
    label.append(0)

for i,image in enumerate(yes_tumor):
    image = cv2.imread(str(image))
    image = Image.fromarray(image,'RGB')
    image = image.resize((64,64))
    dataset.append(np.array(image))
    label.append(1)    

dataset= np.array(dataset)
label =np.array(label)


# Splitting of Dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(dataset,label,test_size=0.2)

#print(X_train.shape)

#Normalizing the images 
x_train = keras.utils.normalize(X_train,axis=1 ) 
x_test = keras.utils.normalize(X_test,axis =1)


# Model (CNN)

model = keras.Sequential([
    
    keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),activation="relu",input_shape =(64,64,3)),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation="relu"),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(100,activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1,activation="sigmoid"),

])

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))

model.save('BrainTumor10epochs.h5')