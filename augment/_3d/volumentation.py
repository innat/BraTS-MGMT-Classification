# -*- coding: utf-8 -*-

import tensorflow as tf 
from functools import partial
from volumentations import *
from config import *

def get_augmentation(patch_size):
    return Compose([
        Rotate((-5, 5), (0, 0), (0, 0), p=0.5),
        RandomCropFromBorders(crop_value=0.1, p=0.3),
        ElasticTransform((0, 0.15), interpolation=2, p=0.5),
        Resize(patch_size, interpolation=1, always_apply=True, p=1.0),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        RandomRotate90((0, 1), p=0.6),
        GaussianNoise(var_limit=(0, 5), p=0.5),
        RandomGamma(gamma_limit=(0.5, 1.5), p=0.7),
    ], p=1.0)

volume3D = get_augmentation((input_height, input_width, input_depth))

def volume3Dfn(image):    
    aug_data = volume3D(**{"image":image})
    return tf.cast(aug_data["image"], tf.float32)

def volumentations_aug(image, label):
    # Wraps a python function and uses it as a TensorFlow op.
    aug_img = tf.numpy_function(func=volume3Dfn, 
                                inp=[image], 
                                Tout=tf.float32)
    aug_img.set_shape((input_height, input_width, input_depth, input_channel))
    return aug_img, label