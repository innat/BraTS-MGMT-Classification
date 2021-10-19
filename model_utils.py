# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 23:29:31 2021

@author: innat
"""
from tensorflow.keras import callbacks 
from config import *

# A custom lr sched 
def get_lr_callback(batch_size=8):
    lr_start   = 0.000005
    lr_max     = 0.00000125 * batch_size
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    return callbacks.LearningRateScheduler(lrfn, verbose=True)


checkpoint_cb = callbacks.ModelCheckpoint(
    filepath=f"model_{fold}.h5", 
    monitor='val_auc', mode='max', 
    save_best_only=True, verbose=1,
    save_weights_only=True
)