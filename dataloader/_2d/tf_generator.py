# -*- coding: utf-8 -*-

import sys 
sys.path.append('D:\Research Zone\BrainTumorClassification')
from config import *
from augment._2d.keras_augmentation import keras_augment, val_resizes, rescale


class TFDataGenerator:
    def __init__(self, data, 
                 shuffle, aug_lib,
                batch_size, rescale):
        self.data = data                  # data files 
        self.shuffle = shuffle            # true for training 
        self.aug_lib = aug_lib            # type of augmentation library 
        self.batch_size = batch_size      # batch size number 
        self.rescale = rescale            # normalize or not 
        
    # a convinient function to get 3D data set 
    def get_3D_data(self):
        if self.aug_lib == 'keras' and self.shuffle:
            self.data = self.data.batch(self.batch_size, drop_remainder=self.shuffle) 
            self.data = self.data.map(lambda x, y: (keras_augment(x), y), num_parallel_calls=AUTO)
        else:
            # true for evaluation and inference, no augmentation 
            self.data = self.data.batch(self.batch_size, drop_remainder=self.shuffle)
            self.data = self.data.map(lambda x, y: (val_resizes(x), y), num_parallel_calls=AUTO)
         
        # rescaling the data for faster convergence 
        if self.rescale:  
            self.data = self.data.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTO)
            
        # prefetching the data 
        return self.data.prefetch(AUTO) 