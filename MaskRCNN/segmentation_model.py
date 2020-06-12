import argparse
import colorsys
import os
import os.path
import pickle
import random
import time
import warnings
from os import listdir

import imutils
import numpy as np
import skimage.io
import tensorflow
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from numpy import asarray, zeros

import cv2

from MaskRCNN.Mask_RCNN.mrcnn import model as modellib
from MaskRCNN.Mask_RCNN.mrcnn import visualize
from MaskRCNN.Mask_RCNN.mrcnn.config import Config
from MaskRCNN.Mask_RCNN.mrcnn.model import MaskRCNN
from MaskRCNN.Mask_RCNN.mrcnn.utils import Dataset
from tensorflow import keras
from keras.models import load_model
#import times



warnings.filterwarnings("ignore")

# PATH_ABSOLUTE = os.path.join(os.getcwd(),'Deepfashion2')
# PATH_VAL_IMAGE = os.path.join(PATH_ABSOLUTE, 'validation', 'image')

class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_config"
 
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # number of classes (we would normally add +1 for the background)
    NUM_CLASSES = 1+1
   
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    
    # Learning rate
    LEARNING_RATE=0.001
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # setting Max ground truth instances
    MAX_GT_INSTANCES=10

    BACKBONE="resnet50"




# print(bboxes)
