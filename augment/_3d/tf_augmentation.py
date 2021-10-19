# -*- coding: utf-8 -*-

import tensorflow as tf 
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.layers.experimental import preprocessing as keras_aug


# https://github.com/tensorflow/models
def equalize(image, mode='grayscale'):
    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = tf.cast(im[..., c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(
            tf.equal(step, 0), lambda: im,
            lambda: tf.gather(build_lut(histo, step), im))
        return tf.cast(result, tf.uint8)

    if mode == 'grayscale':
        image = scale_channel(image, 0)
        return tf.cast(image, tf.float32)
    elif mode == 'rgb':
        s1 = scale_channel(image, 0)
        s2 = scale_channel(image, 1)
        s3 = scale_channel(image, 2)
        image = tf.stack([s1, s2, s3], -1)
        return tf.cast(image, tf.float32)

def tf_image_augmentation(image):  
    # splitted based on modalites. we have 4 type of modalities. Input shape (h, w, depth, channel==4)
    if modeling_in == '3D':
        # input shape: image.shape -> (h, w, input_depth, input_channel)
        # in such condition, we split based on the number of channel, which is 4 here. 
        # after that the variable will contains 4 splitted tensor 
        # each of which is in a shape of (h, w, input_depth, 1)
        splitted_modalities = tf.split(tf.cast(image, tf.float32), input_channel, axis=-1) 
    elif modeling_in == '2D':
        # input shape: image.shape -> (h, w, input_depth * input_channel)
        # in such condition, we split based on the number of channel, which is 4 here.
        # after that the variable will contains 4 splitted tensor 
        # each of which is in a shape of (h, w, input_depth)
        splitted_modalities = tf.split(tf.cast(image, tf.float32), input_channel, axis=-1) 
    
    # augmented frames for 2d modeling 
    # for 2d modeling we use 1 container to gather all augmented samples from 4 modalites
    # however, same augmentation is ensured for each modality for one study
    augment_img = []
    
    # augmented frames for 3d modeling 
    # for 3d modeling we use 4 container to gather all augmented samples from 4 modalites
    # however, same augmentation is ensured for each modality for one study
    flair_augment_img = []
    t1w_augment_img = []
    t1wce_augment_img = []
    t2w_augment_img = []
    
    
    if modeling_in == '3D':
        # remove the last axis.
        # input: (h, w, input_depth, 1) : output: (h, w, input_depth)
        splitted_modalities = [tf.squeeze(i, axis=-1) for i in splitted_modalities] 
    
    # iterate over each modalities, e.g: flair, t1w, t1wce, t2w
    for j, modality in enumerate(splitted_modalities):
        # now splitting each frame from one modality 
        splitted_frames = tf.split(tf.cast(modality, tf.float32), modality.shape[-1], axis=-1)
        
        # iterate over each frame to conduct same augmentation on 
        # each frame 
        for i, img in enumerate(splitted_frames):
            # Given the same seed, they return the same results independent of 
            # how many times they are called.
            # It's very important to get deterministic augmentation results of each modality. 
            tf.random.set_seed(j)
            np.random.seed(j)
            
            # In tf.image.stateless_random_* , the seed is a Tensor of shape (2,) -
            # - whose values are any integers.
            img = tf.image.stateless_random_flip_left_right(img, seed = (j, 2))
            img = tf.image.stateless_random_flip_up_down(img, seed = (j, 2))
            img = tf.image.stateless_random_contrast(img, 0.3, 0.7, seed = (j, 2))
            img = tf.image.stateless_random_brightness(img, 0.5, seed = (j, 2))
            
            # For some operation, it requires channel == 3 
            img = tf.image.stateless_random_saturation(tf.image.grayscale_to_rgb(img), 
                                                       0.9, 1.3, seed = (j, 2))
            img = tf.image.stateless_random_hue(img, 0.3, seed = (j, 2))

            # For some operation we don't need channel == 3, just 1 is enough 
            img = tf.image.rgb_to_grayscale(img)
            img = tf.cast(
                tf.image.stateless_random_jpeg_quality(
                    tf.cast(img, tf.uint8), 
                    min_jpeg_quality=10, max_jpeg_quality=20, seed = (j, 2)
                ), tf.float32)

            # Ensuring same augmentation for each modalities 
            if tf.random.uniform((), seed=j) > 0.8:
                kimg = np.random.choice([1,2,3,4])
                kgma = np.random.choice([0.7, 0.9, 1.2])
                
                # random rate to any of 90, 180, 270, 360 
                img = tf.image.rot90(img, k=kimg) 
                # adjust the gamma
                img = tf.image.adjust_gamma(img, gamma=kgma)  
                
                # additive gaussian noise to image
                noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.2,
                                         dtype=tf.float32, seed=j) 
                img = img + noise  
                

            # The mask_size should be divisible by 2. 
            if tf.random.uniform((), seed=j) > 0.6:
                img = tfa.image.random_cutout(tf.expand_dims(img, 0),
                                              mask_size=(round(input_height * 0.1),
                                                         round(input_width * 0.1)), 
                                              constant_values=0) 
                img = tf.squeeze(img, axis=0)
            
            # additive equalization 
            img = equalize(img, mode='grayscale') 
 
            # Gathering all frames 
            if modeling_in == '3D':
                if j == 0: # 1st modality 
                    flair_augment_img.append(img)
                elif j == 1: # 2nd modality 
                    t1w_augment_img.append(img)
                elif j == 2: # 3rd modality 
                    t1wce_augment_img.append(img)
                elif j == 3:  # 4th modality 
                    t2w_augment_img.append(img)
            elif modeling_in == '2D':
                augment_img.append(img)
      
    
    if modeling_in == '3D':
        image = tf.transpose([flair_augment_img, t1w_augment_img, 
                              t1wce_augment_img, t2w_augment_img])
        image = tf.reshape(image, [input_height, input_width, 
                                   input_depth, input_channel])
    elif modeling_in == '2D':
        image = tf.concat(augment_img, axis=-1)
        image = tf.reshape(image, [input_height, input_width, 
                                   input_depth*input_channel])
    return image