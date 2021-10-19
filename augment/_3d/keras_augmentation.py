# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 19:17:06 2021

@author: innat
"""

from tensorflow.keras import backend as K
from tensorflow.keras.layers.experimental import preprocessing as keras_aug
import tensorflow as tf

import sys 
sys.path.append('D:\Research Zone\BrainTumorClassification')
from config import *
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.layers.experimental import preprocessing as keras_aug


set_seed = None
rand_flip = keras_aug.RandomFlip("horizontal_and_vertical", seed=set_seed)
rand_tran = keras_aug.RandomTranslation(height_factor=0.1, width_factor=0.2, seed=set_seed)
rand_rote = keras_aug.RandomRotation(factor=0.01, seed=set_seed)
rand_cntr = keras_aug.RandomContrast(factor=0.3, seed=set_seed)
rand_crop = keras_aug.RandomCrop(int(input_height*0.97), int(input_width*0.97), seed=set_seed)


def keras_augment(image, label):
    all_modality = tf.reshape(image, [-1, input_height, input_width, 
                                      input_depth*input_channel])
    
    def apply_augment(x):
        x = rand_flip(x)
        x = rand_tran(x)
        x = rand_rote(x)
        x = rand_cntr(x)
        x = rand_crop(x)
        x  = keras_aug.Resizing(input_height, input_width)(x)
        # the following two gives error for some reason, find later 
#         x  = keras_aug.RandomHeight(factor=(0.1, 0.1), seed=set_seed)(x)
#         x  = keras_aug.RandomWidth(factor=(0.1, 0.1), seed=set_seed)(x)
        return x 
    
    aug_images = apply_augment(all_modality)
    image = tf.reshape(aug_images, 
                       [-1, input_height, input_width, 
                        input_depth, input_channel])
    return image, label


# Custom Layer 2 
class RandomEqualize(tf.keras.layers.Layer):
    def __init__(self, prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob
        
    def call(self, inputs, training=True):
        if tf.random.uniform([]) < self.prob:
            return tf.cast(tfa.image.equalize(inputs) / 255., dtype=tf.float32)
        else: 
            return tf.cast(inputs, dtype=tf.float32)
        
    def get_config(self):
        config = {
            'prob': self.prob,
        }
        base_config = super(RandomEqualize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape
        
# Custom Layer 3
class RandomCutout(tf.keras.layers.Layer):
    def __init__(self, prob=0.5, mask_size=(20, 20), replace=1, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob 
        self.replace = replace
        self.mask_size = mask_size
        
    def call(self, inputs, training=True):
        # tf.random.set_seed(seed)
        if tf.random.uniform([]) < self.prob:
            inputs = tfa.image.random_cutout(inputs,
                                             mask_size=self.mask_size,
                                             constant_values=self.replace)  
            return tf.cast(inputs, dtype=tf.float32)
        else: 
            return tf.cast(inputs, dtype=tf.float32)
        
    def get_config(self):
        config = {
            'prob': self.prob,
            'replace': self.replace,
            'mask_size': self.mask_size
        }
        base_config = super(RandomCutout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape  

        