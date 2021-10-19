

import tensorflow as tf 
import numpy as np 
import pydicom, glob, cv2, os
from config import *


# Function to read dicom file 
def read_dicom_xray_and_normalize(path=None, voxel=None):
    if voxel is None:
        dicom = pydicom.read_file(path)
        data  = dicom.pixel_array
    else:
        data = voxel
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def load_voxel(study_id, scan_type="FLAIR", split="train", sz=256):
    data_root = Path(f"../input/rsna-miccai-voxel-{sz}-dataset")
    npy_path = Path(data_root).joinpath("voxel", split, study_id, f"{scan_type}.npy")
    voxel = np.load(str(npy_path))
    return voxel

class BrainTSGeneratorRaw(tf.keras.utils.Sequence):
    def __init__(self, dicom_path, data, split='train'):
        self.data = data
        self.dicom_path = dicom_path
        self.split = split
        self.label = self.data['MGMT_value']
  
    def __len__(self):
        return len(self.data['BraTS21ID'])
    
    def __getitem__(self, index):
        patient_ids = f"{self.dicom_path}/{str(self.data['BraTS21ID'][index]).zfill(5)}/"
   
        # for 3D modeling 
        flair = []
        t1w   = []
        t1wce = []
        t2w   = [] 
        
        # Iterating over each modality
        for m, t in enumerate(input_modality):
            t_paths = sorted(
                glob.glob(os.path.join(patient_ids, t, "*")), 
                key=lambda x: int(x[:-4].split("-")[-1]),
            )
            
            # Pick input_depth times slices -
            # - from middle range possible 
            strt_idx = (len(t_paths) // 2) - int(input_depth / 2)
            end_idx = (len(t_paths) // 2) + int(input_depth / 2)
            # slicing extracting elements with 1 intervals 
            picked_slices = t_paths[strt_idx : end_idx : 1]
            
            # Iterating over the picked slices and do some basic processing, 
            # such as removing black borders
            # and lastly, bind them all in the end.
            for i in picked_slices:
                # Reading pixel file from dicom file 
                image = self.read_dicom_xray(i)
                
                # It's possible that among picked_slices, there can be some black image, 
                # which is not wanted, so we iterate back to dicom file to get - 
                # - any non-black image otherwise move on with black image 
                j = 0
                while True:
                    # if it's a black image, try to pick any random slice of non-black  
                    # otherwise move on with black image. 
                    if image.mean() == 0:
                        # do something 
                        image = self.read_dicom_xray(random.choice(t_paths)) 
                        j += 1
                        if j == 10:
                            break
                    else:
                        break
                        
                # Now, we remove black areas; remove black borders from brain image 
                rows = np.where(np.max(image, 0) > 0)[0]
                cols = np.where(np.max(image, 1) > 0)[0]
                if rows.size:
                    image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
                else:
                    image = image[:1, :1]
           
                # In 3D modeling, we now add frames / slices of individual modalities 
                if m == 0:
                    # Adding flair 
                    flair.append(cv2.resize(image, (input_height, input_width)))
                elif m == 1:
                    # Adding t1w
                    t1w.append(cv2.resize(image, (input_height, input_width)))
                elif m == 2:
                    # Adding t1wce
                    t1wce.append(cv2.resize(image, (input_height, input_width)))
                elif m == 3:
                    # Adding t2w
                    t2w.append(cv2.resize(image, (input_height, input_width)))
                    
        # [ATTENTION!!!]
        # input_shape: (None, h, w, depth, channel)
        # It's possible that with current data loader set up, 
        # All modalities MAY NOT HAVE SAME number of slices.
        # In that case, we adopt some workaround.
        # Right now, we re-append the existing slice with small color variation.
        # Just to avoid this issue.
        # for flair 
        while True:
            if len(flair) < input_depth and flair:
                flair.append(cv2.convertScaleAbs(random.choice(flair), 
                                                 alpha=1.2, beta=0))
            else:
                break
        # for t1w
        while True:
            if len(t1w) < input_depth and t1w:
                t1w.append(cv2.convertScaleAbs(random.choice(t1w), 
                                               alpha=1.1, beta=0))
            else:
                break
        # for t1wce
        while True:
            if len(t1wce) < input_depth and t1wce:
                t1wce.append(cv2.convertScaleAbs(random.choice(t1wce), 
                                                 alpha=1.2, beta=0))
            else:
                break
        # for t2w
        while True:
            if len(t2w) < input_depth and t2w:
                t2w.append(cv2.convertScaleAbs(random.choice(t2w), 
                                               alpha=1.1, beta=0))
            else:
                break

        flair_x = np.moveaxis(np.array(flair), 0, -1) 
        t1_x    = np.moveaxis(np.array(t1w), 0, -1)   
        t1w_x   = np.moveaxis(np.array(t1wce), 0, -1) 
        t2_x    = np.moveaxis(np.array(t2w), 0, -1)   
        
        if self.split == 'train':
            if np.random.rand() < 0.2:
                flair_x, t1_x = t1_x, flair_x
            elif np.random.rand() < 0.4:
                t1w_x,   t2_x = t2_x, t1w_x
            elif np.random.rand() < 0.6:
                t2_x,    flair_x = flair_x, t2_x
            else:
                pass 
        
        return {
            'flair' : flair_x, 
            't1'    : t1_x, 
            't1w'   : t1w_x, 
            't2'    : t2_x 
        }, self.label.iloc[index,]
            
    # Function to read dicom file 
    def read_dicom_xray(self, path):
        data = pydicom.read_file(path).pixel_array
        if data.mean() == 0:
            # If all black, return data and find non-black if possible.
            return data 
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        return data
    
    
    
    
class BrainTSGeneratorRegistered(tf.keras.utils.Sequence):
    def __init__(self, dicom_path, data, split='train'):
        self.data = data
        self.dicom_path = dicom_path
        self.label = self.data['MGMT_value']
        self.split = split
        
    def __len__(self):
        return len(self.data['BraTS21ID'])
    
    def __getitem__(self, index):
        patient_ids = f"{str(self.data['BraTS21ID'][index]).zfill(5)}/"
       
        voxel = load_voxel(patient_ids, input_modality[0], "train")
        voxel = voxel[len(voxel) // 2 - int(input_depth / 2): (len(voxel) // 2) + int(input_depth / 2)]
        voxel = read_dicom_xray_and_normalize(path=None, voxel=voxel)
        flair_x = np.moveaxis(voxel, 0, -1)
        flair_x = keras_aug.Resizing(input_height, input_width)(flair_x)
        
        voxel = load_voxel(patient_ids, input_modality[1], "train")
        voxel = voxel[len(voxel) // 2 - int(input_depth / 2): (len(voxel) // 2) + int(input_depth / 2)]
        voxel = read_dicom_xray_and_normalize(path=None, voxel=voxel)
        t1_x  = np.moveaxis(voxel, 0, -1)
        t1_x  = keras_aug.Resizing(input_height, input_width)(t1_x)
        
        voxel = load_voxel(patient_ids, input_modality[2], "train")
        voxel = voxel[len(voxel) // 2 - int(input_depth / 2): (len(voxel) // 2) + int(input_depth / 2)]
        voxel = read_dicom_xray_and_normalize(path=None, voxel=voxel)
        t1w_x = np.moveaxis(voxel, 0, -1)
        t1w_x = keras_aug.Resizing(input_height, input_width)(t1w_x)
        
        voxel = load_voxel(patient_ids, input_modality[3], "train")
        voxel = voxel[len(voxel) // 2 - int(input_depth / 2): (len(voxel) // 2) + int(input_depth / 2)]
        voxel = read_dicom_xray_and_normalize(path=None, voxel=voxel)
        t2_x  = np.moveaxis(voxel, 0, -1)
        t2_x  = keras_aug.Resizing(input_height, input_width)(t2_x)
        
        if self.split:
            if np.random.rand() < 0.2:
                flair_x, t1_x = t1_x, flair_x
            elif np.random.rand() < 0.4:
                t1w_x,   t2_x = t2_x, t1w_x
            elif np.random.rand() < 0.6:
                t2_x,    flair_x = flair_x, t2_x
            else:
                pass 
        
        return {
            'flair' : flair_x, 
            't1'    : t1_x, 
            't1w'   : t1w_x, 
            't2'    : t2_x 
        }, self.label.iloc[index,] 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    