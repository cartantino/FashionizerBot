import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet18 import preprocess_input

from sklearn import svm
from sklearn.svm import SVC
import keras
from classification_models.keras import Classifiers

import pickle
import cv2

# per modelli dalla libreria classification_models
# ResNet18, preprocess_input = Classifiers.get('resnet18')
# base_model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=False)
# output = keras.layers.GlobalAveragePooling2D()(base_model.output)
# model = keras.models.Model(inputs=[base_model.input], outputs=[output])

# per modelli direttamente da keras.applications
base_model = ResNet50(input_shape=(224,224,3), weights = 'imagenet', include_top = False)
output = GlobalAveragePooling2D()(base_model.output)
model = Model(inputs=base_model.input, outputs=output)


