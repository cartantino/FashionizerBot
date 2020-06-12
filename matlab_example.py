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
from tensorflow.keras.applications.resnet50 import preprocess_input

from MaskRCNN.Mask_RCNN.mrcnn import visualize
from Updater import Updater



def fileparts(fn):
    (dirName, fileName) = os.path.split(fn)
    (fileBaseName, fileExtension) = os.path.splitext(fileName)
    return dirName, fileBaseName, fileExtension


def textHandler(bot, message, chat_id, text):
	if(text == 'yes' or text == 'Yes'):
		bot.sendMessage(chat_id, "Retrieving images..")
	else:
		bot.sendMessage(chat_id, "I do not understand what you said")
	return text


def imageHandler(bot, message, chat_id, local_filename, name):
    print("Filename = " + local_filename)
    # send message to user
    bot.sendMessage(chat_id, "Hi " + name)
    bot.sendMessage(chat_id, "Please, wait a few seconds while I elaborate your image..")
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    print("Current dir = " + cur_dir)
    dirName, fileBaseName, fileExtension = fileparts(local_filename)

    model = bot.getSegmentationModel()

    image = skimage.io.imread(local_filename)
    # Run detection
    results = model.detect([image], verbose=1)
    # Visualize results

    r = results[0]
    bboxes = r['rois']
    print(bboxes)
    print(r['class_ids'])

    # Retrieve 
    sub_images = []
    for bb in bboxes:
        print(bb)
        cropped = image[bb[0]:bb[2], bb[1]:bb[3]]
        cropped = preprocess_input(cropped)
        sub_images.append(cropped)



    #['no clothes','clothes']
    print(range(len(bboxes)))

    r['class_ids'] = np.array([a for a in range(len(bboxes))])

    class_ids = []
    for id_bb in range(len(bboxes)):
        class_ids.append('bbox_' + str(id_bb))

    #class_ids = np.array(class_ids)
    #
    # print(class_ids)


    visualize.display_instances(image, bboxes, r['masks'], r['class_ids'], class_ids , r['scores'],  save_dir=dirName, img_name=fileBaseName + "_ok" + fileExtension)
  
    



    # send back the manipulated image
    new_fn = os.path.join(dirName, fileBaseName + '_ok' + fileExtension)
    bot.sendImage(chat_id, new_fn, "")
    print("Image sent")

    #bot.sendMessage(chat_id, "Would you like to retrieve most similar dresses i know?")



if __name__ == "__main__":
	bot_id = '1116447517:AAFIDT7Efa6-ULbi9wUZPT7lGyzm-Jxdp9s'
	updater = Updater(bot_id)
	updater.setPhotoHandler(imageHandler)
	updater.setTextHandler(textHandler)
	updater.start()
