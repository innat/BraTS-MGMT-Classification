# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 11:19:47 2021

@author: innat
"""

from augment.volumentation import *
from config import *

class TFDataGenerator:
    def __init__(self, data, modeling_in, 
                 shuffle, aug_lib,augmentor,
                batch_size, rescale):
        self.data = data                  # data files 
        self.modeling_in = modeling_in    # 2D or 3D 
        self.shuffle = shuffle            # true for training 
        self.aug_lib = aug_lib            # type of augmentation library 
        self.augmentor = augmentor
        self.batch_size = batch_size      # batch size number 
        self.rescale = rescale            # normalize or not 
        
    # a convinient function to get 3D data set 
    def get_3D_data(self):
        # augmentation on 3D data set
        # volumentation is based on albumentation 
        
        if self.aug_lib == 'volumentations' and self.shuffle:
            # if true, augmentation would be applied separately for each depth 
            self.data = self.data.map(partial(self.augmentor), num_parallel_calls=AUTO)
            self.data = self.data.batch(batch_size, drop_remainder=self.shuffle)
            
        elif self.aug_lib == 'tf' and self.shuffle:
            self.data = self.data.map(lambda x, y: (self.augmentor(x), y),
                                      num_parallel_calls=AUTO).batch(self.batch_size,
                                                                     drop_remainder=self.shuffle)
        elif self.aug_lib == 'keras' and self.shuffle:
                self.data = self.data.batch(self.batch_size, drop_remainder=self.shuffle) 
                self.data = self.data.map(partial(self.augmentor), num_parallel_calls=AUTO)
        else:
            # true for evaluation and inference, no augmentation 
            self.data = self.data.batch(self.batch_size, drop_remainder=self.shuffle)
        
        # rescaling the data for faster convergence 
        if self.rescale:    
            self.data = self.data.map(lambda x, y: (K.clip(x, min_value=0., max_value=255.), y), 
                                      num_parallel_calls=AUTO)
            self.data = self.data.map(lambda x, y: (keras_aug.Rescaling(scale=1./255, offset=0.0)(x), y), 
                                      num_parallel_calls=AUTO)
            
        # prefetching the data 
        return self.data.prefetch(AUTO) 