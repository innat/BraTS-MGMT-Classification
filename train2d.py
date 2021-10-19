# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 22:46:15 2021

@author: innat
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os, glob, random, cv2, glob, pydicom
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from config import *

# For reproducible results    
seed_all(global_seed)
accelerate_gpu(mixed_precision)

# these are corrupted id, so here we are just removing them
df = pd.read_csv(train_df_path)
df = df[~df.BraTS21ID.isin([109, 709, 123])]
df = df.reset_index(drop=True)


skf = StratifiedKFold(n_splits=num_of_fold, shuffle=True, random_state=global_seed)
for index, (train_index, val_index) in enumerate(skf.split(X=df.index, y=df.MGMT_value)):
    df.loc[val_index, 'fold'] = index
print(df.groupby(['fold', df.MGMT_value]).size())

from dataloader._2d.sample_loader import BrainTSGeneratorRegistered, BrainTSGeneratorRaw


def fold_generator(fold):
    
    train_labels = df[df.fold != fold].reset_index(drop=True)
    val_labels   = df[df.fold == fold].reset_index(drop=True)
    
    return (
        BrainTSGeneratorRaw(trian_img_path, train_labels, split='train'), 
        BrainTSGeneratorRaw(trian_img_path, val_labels,  split='validation')
    )

# Get fold set
train_gen, val_gen = fold_generator(fold)

train_data = tf.data.Dataset.from_generator(
    lambda: map(tuple, train_gen),
    (
        {
            'flair' : tf.float32, 
            't1'    : tf.float32,
            't1w'   : tf.float32, 
            't2'    : tf.float32},
        tf.float32),
    (
        {
            'flair' : tf.TensorShape([None, None, None]),
            't1'    : tf.TensorShape([None, None, None]),
            't1w'   : tf.TensorShape([None, None, None]),
            't2'    : tf.TensorShape([None, None, None])},
        tf.TensorShape([]),
    ),
)

val_data = tf.data.Dataset.from_generator(
    lambda: map(tuple, val_gen),
    (
        {
            'flair' : tf.float32, 
            't1'    : tf.float32,
            't1w'   : tf.float32, 
            't2'    : tf.float32},
        tf.float32),
    (
        {
            'flair' : tf.TensorShape([None, None, None]),
            't1'    : tf.TensorShape([None, None, None]),
            't1w'   : tf.TensorShape([None, None, None]),
            't2'    : tf.TensorShape([None, None, None])},
        tf.TensorShape([]),
    ),
)


if aug_lib == 'keras' :
    from augment._2d.keras_augmentation import *
    augmentor = keras_augment 
else:
    augmentor = None



from dataloader._2d.tf_generator import TFDataGenerator

tf_gen = TFDataGenerator(train_data,
                         shuffle=True,     
                         aug_lib='keras', 
                         batch_size=batch_size,   
                         rescale=True
                        )     
train_generator = tf_gen.get_3D_data()


tf_gen = TFDataGenerator(val_data,
                         shuffle=False,     
                         aug_lib=None,    
                         batch_size=batch_size,   
                         rescale=True    
                        ) 
valid_generator = tf_gen.get_3D_data()


x, y = next(iter(train_generator))
a = x['flair'] 
b = x['t1']    
c = x['t1w']  
d = x['t2']    
print(a.shape, y.shape, a.numpy().max(), a.numpy().min())  
print(b.shape, y.shape, b.numpy().max(), b.numpy().min())  
print(c.shape, y.shape, c.numpy().max(), c.numpy().min())  
print(d.shape, y.shape, d.numpy().max(), d.numpy().min())  

for j, (x, y) in enumerate(train_generator.take(1)):
    a = x['flair'] ; print(a.shape)
    b = x['t1']    ; print(b.shape)
    c = x['t1w']   ; print(c.shape)
    d = x['t2']    ; print(d.shape)
    
    for m in [a, b, c, d]:
        plt.figure(figsize=(25, 25))
        for i in range(input_depth):
            if y[0].numpy() != 0 and j != 0:
                continue 
            plt.subplot(8, 8, i + 1)
            plt.imshow(m[0 ,:, :, i].numpy().astype('float32'), cmap='gray')
            plt.title(y[0].numpy())
            plt.axis("off")
        plt.show()
        print('\n'*2)
        
        
from model._2d.classifier import get_model 

tf.keras.backend.clear_session()
model = get_model(summary=True, plot=False)

import tensorflow_addons as tfa
from model_utils import get_lr_callback, checkpoint_cb


lr = get_lr_callback(batch_size)

#Optimizers 
# tf.keras.optimizers.SGD(0.01)
opt = tf.keras.optimizers.Adam(0.03) # tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
opt_ma = tfa.optimizers.MovingAverage(opt)
opt_swa = tfa.optimizers.SWA(opt)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.01),
    optimizer=opt,
    metrics=[tf.keras.metrics.AUC(name='auc'), 
             tf.keras.metrics.FalsePositives(name='fp'),
             tf.keras.metrics.FalseNegatives(name='fn'),
             tf.keras.metrics.BinaryAccuracy(name='acc')],
)

import psutil
model.fit(
    train_generator,
    epochs=1,
    validation_data=valid_generator,
    workers=psutil.cpu_count(),
    callbacks=[lr, checkpoint_cb],
    verbose=1
)



















