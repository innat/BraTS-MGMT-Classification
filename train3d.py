import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import config

# For reproducible results    
config.seed_all(config.global_seed)
config.accelerate_gpu(config.mixed_precision)

# these are corrupted id, so here we are just removing them
train_df_path, trian_img_path, registered_samples = config.sample_path(registered_samples = False)
df = pd.read_csv(train_df_path)
df = df[~df.BraTS21ID.isin([109, 709, 123])]
df = df.reset_index(drop=True)


skf = StratifiedKFold(n_splits=config.num_of_fold, 
                      shuffle=True, 
                      random_state=config.global_seed)
for index, (train_index, val_index) in enumerate(skf.split(X=df.index, y=df.MGMT_value)):
    df.loc[val_index, 'fold'] = index
print(df.groupby(['fold', df.MGMT_value]).size())



from dataloader._3d.sample_loader import BrainTSGeneratorRegistered, BrainTSGeneratorRaw


def fold_generator(fold):
    train_labels = df[df.fold != fold].reset_index(drop=True)
    val_labels   = df[df.fold == fold].reset_index(drop=True)
    
    if registered_samples:
        return (
            BrainTSGeneratorRegistered(trian_img_path, train_labels),
            BrainTSGeneratorRegistered(trian_img_path, val_labels)
        )
    else:
        return (
            BrainTSGeneratorRaw(trian_img_path, train_labels),
            BrainTSGeneratorRaw(trian_img_path, val_labels)
        )

# Get fold set
train_gen, val_gen = fold_generator(config.fold)

for x, y in train_gen:
    print(x.shape, y.shape)
    break

train_data = tf.data.Dataset.from_generator(
    lambda: map(tuple, train_gen),
    (tf.float32, tf.float32),
    (
        tf.TensorShape([config.input_height, config.input_width, config.input_depth, 
                        config.input_channel]),
        tf.TensorShape([]),
    ),
)

val_data = tf.data.Dataset.from_generator(
    lambda: map(tuple, val_gen),
    (tf.float32, tf.float32),
    (
        tf.TensorShape([config.input_height, config.input_width, config.input_depth,
                        config.input_channel]),
        tf.TensorShape([]),
    ),
)


from augment._3d.keras_augmentation import *
from augment._3d.tf_augmentation import * 
from augment._3d.volumentation import * 

if config.aug_lib == 'keras' :
    augmentor = keras_augment 
elif config.aug_lib == 'tf':
    augmentor = tf_image_augmentation  
elif config.aug_lib == 'volumentations':
    augmentor = volumentations_aug 
else:
    augmentor = None
    
    
from dataloader._3d.tf_generator import TFDataGenerator
tf_gen = TFDataGenerator(train_data,
                         modeling_in=modeling_in, 
                         shuffle=True,     
                         aug_lib=aug_lib, 
                         augmentor=augmentor,
                         batch_size=batch_size,   
                         rescale=False)     
train_generator = tf_gen.get_3D_data()


x, y = next(iter(train_generator))
print(x.shape, y.shape, x.numpy().max(), y.numpy().min())  

for i, (x, y) in enumerate(train_generator.take(3)):
    if modeling_in == '3D':
        for j in range(input_channel):
            plt.figure(figsize=(30, 20))
            for i in range(input_depth):
                plt.subplot(8, 8, i + 1)
                plt.imshow(x[0 ,:, :, i, j], cmap='gray')
                plt.axis("off")
                plt.title(y[0].numpy())
            plt.show()
        print('\n'*3)
    elif modeling_in == '2D':
        plt.figure(figsize=(30, 20))
        for i in range(input_depth*input_channel):
            plt.subplot(5, 8, i + 1)
            plt.imshow(x[0 ,:, :, i], cmap='gray')
            plt.axis("off")
            plt.title(y[0].numpy())
            
            
tf_gen = TFDataGenerator(val_data,
                         modeling_in=modeling_in, 
                         shuffle=False,     
                         aug_lib=None,    
                         augmentor=None,
                         batch_size=batch_size,   
                         rescale=False    
                        ) 
valid_generator = tf_gen.get_3D_data()
    
    
from model._3d.classifier import get_model 
tf.keras.backend.clear_session()
model = get_model(input_width, input_height, input_depth, input_channel)
model.summary()

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=[tf.keras.metrics.AUC(), 
             tf.keras.metrics.BinaryAccuracy(name='acc')],
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath="model.{epoch:02d}-{val_auc:.4f}.h5", 
    monitor='val_auc', mode='max', 
    save_best_only=True, verbose=1
)

epochs = 2
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=valid_generator,
    callbacks=[checkpoint_cb], verbose=1
)