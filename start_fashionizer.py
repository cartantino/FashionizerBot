import argparse
import colorsys
import os
import os.path
import pickle
import platform
import random
import subprocess
import sys
import time
import warnings
from os import listdir

import cv2
import imutils
import numpy as np
import skimage.io
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy import asarray, zeros
from PIL import Image
from sklearn.svm import SVC
from tensorflow.keras.applications.resnet50 import preprocess_input

from MaskRCNN.Mask_RCNN.mrcnn import visualize
from Updater import Updater


def classify_image(img, model_features):
    with open('classifier/clf_resnet50.pickle', 'rb') as handle:
        classifier = pickle.load(handle)


    dim = (224, 224)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    x = preprocess_input(img)
    x= x.reshape((-1, 224, 224, 3))
    features = model_features.predict(x)
    features = scaler.transform(features)
    pred = classifier.predict(features)
    #prob_pred = classifier.predict_proba(features)

    with open('classifier/labels_resnet50.pickle', 'rb') as handle:
        labels = pickle.load(handle)

    labels = [ labels[key] for key in labels ]
    print(pred[0])
    return labels[pred[0]], 0.09

def classify_image_svm(img, feature_extractor, svm_classifier):
    with open('data/scaler.pickle', 'rb') as handle:
        scaler = pickle.load(handle)

    dim = (224, 224)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    x = preprocess_input(img)
    x = np.expand_dims(x, axis=0)
    neural_features = feature_extractor.predict(x)[0]
    neural_features = neural_features.reshape(len(neural_features),)
    neural_features = [neural_features]
    features = scaler.transform(neural_features)
    pred = svm_classifier.predict(features)
    prob_pred = svm_classifier.predict_proba(features)
    return pred[0], max(max(prob_pred))


# function to get unique values 
def unique(list1): 
    x = np.array(list1) 
    return np.unique(x)


def fileparts(fn):
    (dirName, fileName) = os.path.split(fn)
    (fileBaseName, fileExtension) = os.path.splitext(fileName)
    return dirName, fileBaseName, fileExtension


def textHandler(bot, message, chat_id, text):
	message = "Hello, I am FashionizerBot.. to see what can I do send me an image of someone dressing something cool"
	return message


def imageHandler(bot, message, chat_id, local_filename, name):
    print("Filename = " + local_filename)
    # send message to user
    bot.sendMessage(chat_id, "Hi " + name)
    bot.sendMessage(chat_id, "Please, wait a few seconds while I elaborate your image..")
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    print("Current dir = " + cur_dir)
    dirName, fileBaseName, fileExtension = fileparts(local_filename)

    image = skimage.io.imread(local_filename)

    with open('classifier/SVM_resnet18_neural_features.pickle', 'rb') as handle:
        svm_classifier = pickle.load(handle)
 
    print(svm_classifier)

    feature_extractor = bot.getFeatureExtractorModel()
    preprocess_input_extractor = bot.getExtractorPreprocessing()

    segmentation_model = bot.getSegmentationModel()
    #classification_model = bot.getClassificationModel()

    # Segmentation
    results = segmentation_model.detect([image], verbose=1)

 
    # Visualize results
    r = results[0]
    mask = r['masks']
    bboxes = r['rois']
    #print(bboxes)
    #print(r['class_ids'])
    #print(mask)
  
    
    # Create a new file where we will store the new image
    new_fn = os.path.join(dirName, fileBaseName + '_ok' + fileExtension)


    r['class_ids'] = np.array([a for a in range(len(bboxes))])

    class_ids = []
    for id_bb in range(len(bboxes)):
        class_ids.append('bbox_' + str(id_bb))


    predicted_labels = []
    pred_probabilities = []

    
    for i, bb in enumerate(bboxes):
        current_mask = mask[:,:,i]
        result = image.copy()
        result[current_mask != 0] = 255
        cropped = result[bb[0]:bb[2], bb[1]:bb[3]]
        plt.imshow(cropped)
        cropped = preprocess_input_extractor(cropped)  
        #prediction, probabilities = classify_image(cropped, classification_model)
        prediction, probabilities = classify_image_svm(cropped, feature_extractor, svm_classifier)
        predicted_labels.append(prediction)
        pred_probabilities.append(probabilities)

    
    labels = unique(predicted_labels)
    print(labels)
    #for label in labels:
    #    n_label = predicted_labels.count(label)
    #    bot.sendMessage(chat_id, "I have found " + str(n_label) + " of label")
    if(len(bboxes) == 0):
        bot.sendMessage("Nothing found")
    else:
        visualize.display_instances(image, bboxes, r['masks'], r['class_ids'], predicted_labels , pred_probabilities,  save_dir=dirName, img_name=fileBaseName + "_ok" + fileExtension)
        bot.sendImage(chat_id, new_fn, "")
        print("Elaborated image sent to " + name)

    #print("Predictions = " + str(predicted_labels))
    #print("Probabilities = " + str(pred_probabilities))



if __name__ == "__main__":
    bot_id = '1116447517:AAFIDT7Efa6-ULbi9wUZPT7lGyzm-Jxdp9s'
    print("Creating updater instance..")
    updater = Updater(bot_id)
    updater.setPhotoHandler(imageHandler)
    updater.setTextHandler(textHandler)
    updater.start()
    print("Bot started!")
