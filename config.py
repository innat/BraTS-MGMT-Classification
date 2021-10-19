# Training Fold
num_of_fold = 5
fold = 0
global_seed = 101
modeling_in = '3D' # Support:'2D' or '3D' modeling,  
aug_lib = 'keras'  # Support, arg: 'tf', 'keras', 'volumentations'
input_modality = ["FLAIR","T1w", "T1wCE", "T2w"] 
batch_size    = 3
input_height  = 128
input_width   = 128
input_channel = len(input_modality)  # Total number of channel, e.g. 4 
input_depth   = 24 # Total number of picked slices from each modality, e.g. 30 
mixed_precision = True


train_df_path  = 'D:/Kaggle & DataSets/Kaggle/BrainTumor/train_labels.csv'
trian_img_path = 'D:/Kaggle & DataSets/Kaggle/BrainTumor/train_brain_tumor_kaggle/'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 

# Params 
AUTO = tf.data.AUTOTUNE
print(tf.executing_eagerly())
print('A: ', tf.test.is_built_with_cuda)
print('B: ', tf.test.gpu_device_name())

def accelerate_gpu(mp=False):
    GPUS = tf.config.list_physical_devices('GPU')
    if GPUS:
        try:
            for GPU in GPUS:
                tf.config.experimental.set_memory_growth(GPU, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(GPUS), "Physical GPUs,", len(logical_gpus), "Logical GPUs") 
        except RuntimeError as  RE:
            print(RE)
    if mp:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print('Mixed precision enabled')

import random, numpy as np, os
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(s) 
    
    