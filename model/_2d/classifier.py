# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 21:26:11 2021

@author: innat
"""

import tensorflow as tf
from config import *
from classification_models.tfkeras import Classifiers
from tensorflow.keras import applications as model
from tensorflow.keras import layers 
from tensorflow.keras import Input, Model

def build_model(base_net):
    h, w, c = input_height, input_width, 3
    
    if base_net == 'a':
        base_Mchannel = model.EfficientNetB2(input_shape=(h, w, input_depth),
                                               include_top=False,
                                               weights=None)
        base_Mchannel._layers.pop(1) # remove rescaling
        base_Mchannel._layers.pop(1) # remove normalization
#         base_w_mchannel = tf.keras.models.load_model('../input/demo-grats-2d-binary/dense_model_40depth.h5')
        x = layers.GlobalAveragePooling2D(name='gap_0')(base_Mchannel.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5, name='drop_0')(x)
        return Model(inputs=base_Mchannel.input, outputs=x, name='model_0')
    
    elif base_net == 'b':
        base_Mchannel = model.DenseNet121(input_shape=(h, w, input_depth),
                                                 include_top=False,  
                                                 weights=None)
#         base_w_mchannel = tf.keras.models.load_model('../input/demo-grats-2d-binary/dense_model_40depth.h5')
        x = layers.GlobalAveragePooling2D(name='gap_1')(base_Mchannel.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5, name='drop_1')(x) 
        return Model(inputs=base_Mchannel.input, outputs=x, name='model_1')
        
    elif base_net == 'c':
        get_model, _ = Classifiers.get('resnet18')
        base_Mchannel = get_model(include_top = False, 
                                  input_shape=(h, w, input_depth), 
                                  weights=None)
#         base_Mchannel = tf.keras.models.load_model('../input/demo-grats-2d-binary/dense_model_40depth.h5')
        x = layers.GlobalAveragePooling2D(name='gap_2')(base_Mchannel.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5, name='drop_2')(x) 
        return Model(inputs=base_Mchannel.input, outputs=x, name='model_2')
        
    elif base_net == 'd':
        get_model, _ = Classifiers.get('seresnext50')
        base_Mchannel = get_model(include_top = False, 
                                  input_shape=(h, w, input_depth), 
                                  weights=None)
#         base_w_mchannel = tf.keras.models.load_model('../input/demo-grats-2d-binary/dense_model_40depth.h5')
        x = layers.GlobalAveragePooling2D(name='gap_3')(base_Mchannel.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5, name='drop_3')(x) 
        return Model(inputs=base_Mchannel.input, outputs=x, name='model_3')
    
    

# detect and init the TPU
model_a = build_model('a')
model_b = build_model('b')
model_c = build_model('c')
model_d = build_model('d')

for i, layer in enumerate(model_a.layers):
    if i == 0: 
        layer._name = 'flair'
    else:
        layer._name = layer._name + str("a")

for i, layer in enumerate(model_b.layers):
    if i == 0:
        layer._name = 't1'
    else:
        layer._name = layer._name + str("b")

for i, layer in enumerate(model_c.layers):
    if i == 0:
        layer._name = 't1w'
    else:
        layer._name = layer._name + str("c")

for i, layer in enumerate(model_d.layers):
    if i == 0:
        layer._name = 't2'
    else:
        layer._name = layer._name + str("d")
    
    
print(model_a.output_shape)
print(model_b.output_shape)
print(model_c.output_shape)
print(model_d.output_shape)


x_a = layers.Dense(764, activation='relu')(model_a.output)
x_b = layers.Dense(764, activation='relu')(model_b.output)
x_c = layers.Dense(764, activation='relu')(model_c.output)
x_d = layers.Dense(764, activation='relu')(model_d.output)


concate_models_output = layers.average([x_a, x_b, x_c, x_d])
head_layer = layers.Dense(512, activation='relu')(concate_models_output)
head_layer = layers.Dropout(0.5)(head_layer)
findal_output = layers.Dense(1, activation='sigmoid', dtype=tf.float32)(head_layer)

model = Model(
    inputs  = [model_a.input, model_b.input,
               model_c.input, model_d.input],
    outputs = [findal_output]
)

for i in range(len(model.weights)):
    model.weights[i]._handle_name = model.weights[i].name + "_" + str(i)
    
    
    

from tensorflow.keras import backend as K

def get_model(summary, plot):
    if summary:
        trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
        print('Total params: {:,}'.format(trainable_count + non_trainable_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))
        
    if plot:
        display(tf.keras.utils.plot_model(model, show_layer_names=True))
        
    return model 

    
    
    
    
    
    
    
    
    
    
    
    
    

