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
import sklearn.neighbors
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy import asarray, zeros
from PIL import Image
from sklearn.neighbors import KDTree
from sklearn.svm import SVC

from MaskRCNN.Mask_RCNN.mrcnn import visualize
from Updater import Updater

RETRIEVAL_N_IMAGES = 3


def classify_image(img, model, preprocess_input):
    dim = (224, 224)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    x = preprocess_input(img)
    x = np.expand_dims(x, axis=0)   
    pred = model.predict(x)
    probabilities = pred[0]
    max_prob_index = np.argmax(probabilities,axis=0)
    max_prob = probabilities[max_prob_index]
 
    print(max_prob)
    print(max_prob_index)

    with open('data/labels_resnet18.pickle', 'rb') as handle:
        labels = pickle.load(handle)

    labels = [ labels[key] for key in labels ]

    predicted_label = labels[max_prob_index]
    print(predicted_label)
    #print(pred[0])
    #return labels[pred[0]], 0.09
    return predicted_label, max_prob


def classify_image_svm(img, feature_extractor, svm_classifier, preprocess_input):
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


def image_retrieval(img, feature_extractor, kdTree, preprocess_input, k_neighbors):
    with open('data/filenames.pickle', 'rb') as handle:
        filenames = pickle.load(handle)
    dim = (224, 224)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    x = preprocess_input(img)
    x = np.expand_dims(x, axis=0)
    neural_features = feature_extractor.predict(x)[0]
    neural_features = neural_features.reshape(len(neural_features),)
    neural_features = [neural_features]
    dist, indexes = kdTree.query(neural_features, k = k_neighbors) # return k images per bb
    retrieval_filenames = [filenames[index] for index in indexes[0]]
    distances = dist[0]
    print(distances)
    return distances, retrieval_filenames




# function to get unique values 
def unique(list1): 
    x = np.array(list1) 
    return np.unique(x)


def fileparts(fn):
    (dirName, fileName) = os.path.split(fn)
    (fileBaseName, fileExtension) = os.path.splitext(fileName)
    return dirName, fileBaseName, fileExtension


def textHandler(bot, message, chat_id, text):
    message = "I am FashionizerBot to see what can I do send me an image of someone dressing something cool"
    bot.sendMessage(chat_id, message)
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

    svm_classifier = bot.getSVM()
    resnet_18_finetuning = bot.getResnetFinetuned()
    segmentation_model = bot.getSegmentationModel()
    feature_extractor = bot.getFeatureExtractorModel()
    preprocess_input_extractor = bot.getExtractorPreprocessing()
    KdTree_retrieval = bot.getKdTree()
    print(svm_classifier)

    # Segmentation
    results = segmentation_model.detect([image], verbose=1)

 
    # Visualize results
    r = results[0]
    mask = r['masks']
    bboxes = r['rois']
  
    # Create a new file where we will store the new image
    new_fn = os.path.join(dirName, fileBaseName + '_ok' + fileExtension)


    r['class_ids'] = np.array([a for a in range(len(bboxes))])

    class_ids = []
    for id_bb in range(len(bboxes)):
        class_ids.append('bbox_' + str(id_bb))


    predicted_labels = []
    pred_probabilities = []
    cropped_images = []

    
    for i, bb in enumerate(bboxes):
        current_mask = mask[:,:,i]
        # with open('mask/current_mask.pickle', 'wb') as handle:
        #     pickle.dump(current_mask, handle)
        # result = image.copy()
        # with open('mask/result.pickle', 'wb') as handle:
        #     pickle.dump(result, handle)
        
        #result[current_mask != 1] = 255
        cropped = image[bb[0]:bb[2], bb[1]:bb[3]]

        #cv2.imshow("bbox", cropped)
        #cv2.waitKey(0)
        #cropped = preprocess_input_extractor(cropped)  
        #prediction, probabilities = classify_image(cropped, resnet_18_finetuning, bot.getPreprocessingResnet18())
        prediction, probabilities = classify_image_svm(cropped, feature_extractor, svm_classifier, bot.getPreprocessingResnet18())
        predicted_labels.append(prediction)
        pred_probabilities.append(probabilities)
        cropped_images.append(cropped)



    if(len(bboxes) == 0):
        #bot.sendMessage(chat_id, "Nothing found")
        print("Segmentation failed, looking for something to classify..")
        prediction, probabilities = classify_image(image, resnet_18_finetuning, bot.getPreprocessingResnet18())
        if(probabilities >= 60):
            bot.sendMessage(chat_id, "Found " + prediction + " with " + str(probabilities) + 'of reliability')
            distances, filenames = image_retrieval(image, feature_extractor, KdTree_retrieval, bot.getPreprocessingResnet18(), RETRIEVAL_N_IMAGES)
            bot.sendMessage(chat_id, "These are the " + str(RETRIEVAL_N_IMAGES) + " most similar images to " + prediction)
            for dist, file_ in zip(distances, filenames):
                head, tail = os.path.split(file_.replace('/', '\\'))
                CURRENT_PATH = os.path.join(cur_dir, 'Dataset', 'fashion-dataset','images', tail)
                print("CURRENT PATH = " + CURRENT_PATH)
                bot.sendImage(chat_id, CURRENT_PATH, "")
        else:
            bot.sendMessage(chat_id, "I am working hard to find something but I can't.. can you take a better shot?")
        
    else:
        bbox_colors = visualize.display_instances(image, bboxes, r['masks'], r['class_ids'], predicted_labels , pred_probabilities,  save_dir=dirName, img_name=fileBaseName + "_ok" + fileExtension)
        bot.sendImage(chat_id, new_fn, "")
        print("Elaborated image sent to " + name)
        unique_labels = unique(predicted_labels)

        for label in unique_labels:
            n_label = predicted_labels.count(label)
            bot.sendMessage(chat_id, "I have found " + str(n_label) + " " + label)

        k = RETRIEVAL_N_IMAGES
        for prediction, cropped, color in zip(predicted_labels, cropped_images, bbox_colors):
            distances, filenames = image_retrieval(cropped, feature_extractor, KdTree_retrieval, bot.getPreprocessingResnet18(), k)
            bot.sendMessage(chat_id, "These are the " + str(k) + " most similar images to " + prediction + " found in " + color + " bounding box")
            for dist, file_ in zip(distances, filenames):
                head, tail = os.path.split(file_.replace('/', '\\'))
                CURRENT_PATH = os.path.join(cur_dir, 'Dataset', 'fashion-dataset','images', tail)
                print("CURRENT PATH = " + CURRENT_PATH)
                bot.sendImage(chat_id, CURRENT_PATH, "")
        

    



 



if __name__ == "__main__":
    bot_id = '1116447517:AAFIDT7Efa6-ULbi9wUZPT7lGyzm-Jxdp9s'
    print("Creating updater instance..")
    updater = Updater(bot_id)
    updater.setPhotoHandler(imageHandler)
    updater.setTextHandler(textHandler)
    updater.start()
    print("Bot started!")
