import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf

import numpy as np
import pandas as pd
import os
from glob import glob
import cv2
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
# from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

def decoder(input, nfilters, pool = True):
    x = layers.Conv2D(nfilters,kernel_size = (3, 3),
                  activation = 'relu',
                  padding = 'same', kernel_initializer='he_normal')(input)
    x = layers.Conv2D(nfilters,kernel_size = (3, 3),
                  activation = 'relu',
                  padding = 'same', kernel_initializer='he_normal')(x)
    skip_connection = x
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation('relu')(x)
    if pool:
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    return x, skip_connection


def encoder(input, skip_connection, nfilters):

    input = layers.Conv2DTranspose(
                                nfilters,
                                kernel_size=(2,2), 
                                strides = (2,2), 
                                padding = "same")(input)

    input = layers.concatenate([input, skip_connection], axis=3)

    x = layers.Conv2D(nfilters,kernel_size = (3, 3),
                  activation = 'relu',
                  padding = 'same', kernel_initializer='he_normal')(input)
    x = layers.Conv2D(nfilters,kernel_size = (3, 3),
                  activation = 'relu',
                  padding = 'same', kernel_initializer='he_normal')(input)
    return x

inputs = layers.Input((128, 128, 3))
# layer1 = decoder(inputs, 16)
# layer2 = decoder(layer1[0], 32)
# layer3 = decoder(layer2[0], 64)
# layer4 = decoder(layer3[0], 128)
# layer_middle = decoder(layer4[0], 256 ,pool=False)

# layer6 = encoder(layer_middle[0],layer4[1],128)
# layer7 = encoder(layer6,layer3[1],64)
# layer8 = encoder(layer7,layer2[1],32)
# layer9 = encoder(layer8,layer1[1],16)

layer1 = decoder(inputs, 16)
layer2 = decoder(layer1[0], 32)
layer3 = decoder(layer2[0], 64)

layer_middle = decoder(layer3[0], 128 ,pool=False)

antilayer3 = encoder(layer_middle[0],layer3[1],64)
antilayer2 = encoder(antilayer3,layer2[1],32)
antilayer1 = encoder(antilayer2,layer1[1],16)
res = layers.Conv2D(filters = 1,
                 kernel_size = (1, 1),
                 activation = 'sigmoid',    # use softmax if n_classes>1
                 padding = 'same')(antilayer1)

model = keras.Model(inputs=inputs, outputs=res)
model.summary()
model.load_weights(filepath="unet_brain_mri_seg.hdf5")

def predict_image(path):
    im = cv2.imread(path)
    im = cv2.resize(im, (128,128))
    im = im/255
    data = np.zeros((1,128,128,3))
    data[0] = im

    outputs = model.predict(data)
    prediction = cv2.resize(outputs[0], (256,256))
    prediction = ( prediction >= 0.5).astype(np.uint8)
    cv2.imwrite("out.png", prediction)    
    return "out.png"
