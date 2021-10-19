# -*- coding: utf-8 -*-


import sys 
sys.path.append('D:\Research Zone\BrainTumorClassification')
from tensorflow.keras import backend as K 
from tensorflow.keras.layers.experimental import preprocessing as keras_aug
from config import *

    
set_seed = None 
random_flp       = keras_aug.RandomFlip("horizontal_and_vertical", seed=set_seed)
random_rot       = keras_aug.RandomRotation(factor=0.03, seed=set_seed)
random_translate = keras_aug.RandomTranslation(height_factor=0.1, 
                                               width_factor=0.1, seed=set_seed)
random_contrast  = keras_aug.RandomContrast(factor=0.2, seed=set_seed)
random_crop      = keras_aug.RandomCrop(int(input_height*0.98), 
                                        int(input_width*0.98), seed=set_seed)
resize = keras_aug.Resizing(input_height, input_width)

def keras_augment(image):
    def applying_augment(x):
        x = random_flp(x)
        x = random_rot(x)
        x = random_translate(x)
        x = random_contrast(x)
        x = random_crop(x)
        x = resize(x)
        return x 
    return {
        'flair' :  applying_augment(image['flair']),
        't1'    :  applying_augment(image['t1']),
        't1w'   :  applying_augment(image['t1w']),
        't2'    :  applying_augment(image['t2'])
    }

def val_resizes(image):
    def resizing(x):
        return resize(x)
    return {
        'flair':  resizing(image['flair']),
        't1'   :  resizing(image['t1']),
        't1w'  :  resizing(image['t1w']),
        't2'   :  resizing(image['t2'])
    }

def rescale(image):
    def rescaling(x):
        x = K.clip(x, min_value=0., max_value=255.)
        x = keras_aug.Rescaling(scale=1./255, offset=0.0)(x)
        return x 
    
    return {
        'flair': rescaling(image['flair']),
        't1'   : rescaling(image['t1']),
        't1w'  : rescaling(image['t1w']),
        't2'   : rescaling(image['t2'])
    }