import json
import logging
import os
import pickle
import sys
import tempfile
import time
import urllib

import cv2
import keras
import matplotlib.pyplot as plt
import requests
import tensorflow as tf
from classification_models.keras import Classifiers
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, MaxPooling2D)
from keras.models import Sequential, load_model
from sklearn import svm
from sklearn.svm import SVC
from tensorflow import keras

import MaskRCNN.Mask_RCNN.mrcnn as segmentation_model
from MaskRCNN.Mask_RCNN.mrcnn import model as modellib
from MaskRCNN.Mask_RCNN.mrcnn.utils import Dataset
from MaskRCNN.segmentation_model import (Config, MaskRCNN, myMaskRCNNConfig,
                                         visualize)

#import re, hashlib


def load_segmentation_model():
    print("Settings configuration of the segmentation model...")
    config = myMaskRCNNConfig()
	#Loading the model in the inference mode
    print("Loading model...")
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.path.join(os.getcwd(),"MaskRCNN","mask_rcnn_"))
    print("Loading weights...")
    model.load_weights(os.path.join(os.getcwd(),"MaskRCNN","mask_rcnn_","mask_rcnn_.1591234121.0669577.h5"), by_name=True)
    return model

def load_classification_model():
    print("Loading classification model...")
    #base_model = ResNet50(input_shape=(224,224,3), weights = 'imagenet', include_top = False)
    #output = GlobalAveragePooling2D()(base_model.output)
    #model = Model(inputs=base_model.input, outputs=output)
    model = [1,2,3,4]
    return model

def load_feature_extractor():
    ResNet18, preprocess_input_function = Classifiers.get('resnet18')
    #base_model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=False)
    #output = keras.layers.GlobalAveragePooling2D()(base_model.output)
    #model = keras.models.Model(inputs=[base_model.input], outputs=[output])
    with open(os.path.join(os.getcwd(), 'classifier','resnet18_model.pickle'), 'rb') as handle:
        model = pickle.load(handle)
    return [model, preprocess_input_function]


class Bot:
    def __init__(self, bot_id,  download_folder=tempfile.gettempdir()+os.sep):
        self.bot_id = bot_id
        self.base_url = "https://api.telegram.org/bot" + bot_id + "/"
        self.file_url = "https://api.telegram.org/file/bot" + bot_id + "/"
        self.max_update_id = 0
        self.encoding  = 'utf-8'
        self.download_folder = download_folder
        self.segmentation_model = load_segmentation_model()
        #self.classification_model = load_classification_model()
        feature_extractor_model = load_feature_extractor()
        self.extractor_model = feature_extractor_model[0]
        self.extractor_preprocessing = feature_extractor_model[1]  
        

    def query(self, page, params):
        response = urllib.request.urlopen( self.base_url + page, urllib.parse.urlencode(params).encode(self.encoding) )
        return json.loads(response.read().decode(self.encoding))

    def getSegmentationModel(self):
        return self.segmentation_model

    def getClassificationModel(self):
        return self.classification_model

    def getFeatureExtractorModel(self):
        return self.extractor_model
    
    def getExtractorPreprocessing(self):
        return self.extractor_preprocessing

    def getMessageType(self, message):
        if 'photo' in message:
            return 'photo'
        if 'voice' in message:
            return 'voice'
        if 'document' in message:
            return 'document'
        if 'text' in message:
            return 'text'

    def sendMessage(self, chat_id, text):
        return self.query("sendMessage", { "chat_id": chat_id, "text": text} )
    
    def sendImage(self, chat_id, image_path, caption):
        #http://docs.python-requests.org/en/latest/user/quickstart/
        if os.path.isfile(image_path):
            print("sto inviando l'immagine: " + image_path)
            url = self.base_url + 'sendPhoto'
            files = {'photo': open(image_path, 'rb')}
            data  = {'caption':caption, "chat_id": chat_id}
            r = requests.post(url, files=files, data=data)
        else:
            print("Immagine non trovata: " + image_path)

    def sendDocument(self, chat_id, doc_path):
        #http://docs.python-requests.org/en/latest/user/quickstart/
        if os.path.isfile(doc_path):
            print("sto inviando il documento: " + doc_path)
            url = self.base_url + 'sendDocument'
            files = {'document': open(doc_path, 'rb')}
            data  = {"chat_id": chat_id}
            r = requests.post(url, files=files, data=data)
            #print(r.text)
        else:
            print("Documento non trovato: " + doc_path)

    def waitIfRetrieval(self, chat_id, message):
        if 'text' in message:
            return message


    def getFileDetails(self, file_id):
        file_details = self.query('getFile', { "file_id": file_id})
        #print(file_details)
        #file_path, filename, ext
        file_url   = self.file_url + file_details['result']['file_path']
        file_name  = os.path.basename(file_details['result']['file_path'])
        file_ext   = os.path.splitext(file_name)[1]
        return file_url, file_name, file_ext

    def getFile(self, file_id, download_folder=None):
        if download_folder is None:
            download_folder = self.download_folder
        file_url, file_name, file_ext = self.getFileDetails(file_id)
        local_filename = download_folder + file_name
        urllib.request.urlretrieve(file_url, local_filename)
        return local_filename


    def getUpdates(self, update_id=-1):
        # define which updates to fetch
        if update_id < 0:
            update_id = self.max_update_id + 1
        # get updates
        data = self.query("getUpdates", { "offset": update_id})
        # update max_update_id
        for r in data['result']:
            if r['update_id'] > self.max_update_id:
                self.max_update_id = r['update_id']
        # return updates
        return data['result']




if __name__ == "__main__":
    bot = Bot('128366843:AAHovviK9AQDbcWJkM9JkqDAt8B5oLUUCQI')
    while True:
        #print(bot.getUpdates())
        for u in bot.getUpdates():
            print(u['message'])
            messageType = bot.getMessageType(u['message'])
            print(messageType)
            if 'photo' in u['message']:
                local_filename = bot.getFile(u['message']['photo'][-1]['file_id'])
                print(local_filename)
            if 'voice' in u['message']:
                local_filename = bot.getFile(u['message']['voice']['file_id'])
                print(local_filename)
            if 'document' in u['message']:
                local_filename = bot.getFile(u['message']['document']['file_id'])
                print(local_filename)
        time.sleep(2)
