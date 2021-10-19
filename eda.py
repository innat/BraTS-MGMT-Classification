# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 13:49:14 2021

@author: innat
"""

from config import *

df = pd.read_csv(train_df_path); print(df.shape)
display(df.head()); print(df.MGMT_value.value_counts())

train_sample_path = trian_img_path
len(os.listdir(train_sample_path)), df.BraTS21ID.nunique()

# TODO ..

# x, y = next(iter(train_generator))
# print(x.shape, y.shape, x.numpy().max(), y.numpy().min())  

# Checking 
# augl = Keras3DAugmentation() # augment
# aug_1 = augl(x)
# aug_2 = augl(x)

# plt.imshow(aug_1[0 ,:, :, 0, 0].numpy().astype('float32'), cmap='gray')
# plt.imshow(aug_2[0 ,:, :, 0, 0].numpy().astype('float32'), cmap='gray')
 

# augll = Keras3DAugmentation() # augment
# aug_3 = augll(x)
# aug_4 = augll(x)

# plt.imshow(aug_3[0 ,:, :, 0, 0].numpy().astype('float32'), cmap='gray')
# plt.imshow(aug_4[0 ,:, :, 0, 0].numpy().astype('float32'), cmap='gray')

 
# plt.figure(figsize=(10, 10))
# for j in range(input_channel):
#     plt.figure(figsize=(30, 20))
#     for i in range(input_depth):
#         plt.subplot(8, 8, i + 1)
#         plt.imshow(x[0 ,:, :, i, j], cmap='gray')
#         plt.axis("off")
#         if i != 0: break
#     plt.show()
# print('\n'*2)