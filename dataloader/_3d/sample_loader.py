import tensorflow as tf 
import numpy as np 
import pydicom, glob, cv2, os
from config import *
from pathlib import Path
   
class BrainTSGeneratorRaw(tf.keras.utils.Sequence):
    def __init__(self, dicom_path, data):
        self.data = data
        self.dicom_path = dicom_path
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
            strt_idx = (len(t_paths) // 2) - (input_depth // 2)
            end_idx = (len(t_paths) // 2) + (input_depth // 2)
            # slicing extracting elements with 1 intervals 
            picked_slices = t_paths[strt_idx:end_idx:1]
            
            # Iterating over the picked slices and do some basic processing, 
            # such as removing black borders
            # and lastly, bind them all in the end.
            for i in picked_slices:
                # Reading pixel file from dicom file 
                image = read_dicom_xray_and_normalize(path=i, voxel=None)
                print(image.shape)

                # It's possible that among picked_slices, there can be some black image, 
                # which is not wanted, so we iterate back to dicom file to get - 
                # - any non-black image otherwise move on with black image 
                j = 0
                while True:
                    # if it's a black image, try to pick any random slice of non-black  
                    # otherwise move on with black image. 
                    if image.mean() == 0:
                        # do something 
                        image = read_dicom_xray_and_normalize(path=random.choice(t_paths), voxel=None) 
                        j += 1
                        if j == 100:
                            break
                    else:
                        break
                        
                # Now, we remove black areas; remove black borders from brain image 
                rows = np.where(np.max(image, 0) > 0)[0]
                cols = np.where(np.max(image, 1) > 0)[0]
                if rows.size:
                    image = image[cols[0]: cols[-1] + 1, 
                                  rows[0]: rows[-1] + 1]
                else:
                    image = image[:1, :1]
           
                # In 3D modeling, we now add frames / slices of individual modalities 
                if m == 0:
                    # Adding flair 
                    flair.append(cv2.resize(image, (input_height, input_width)))
                elif m == 1:
                    # Adding t1w
                    t1w.append(cv2.resize(image,   (input_height, input_width)))
                elif m == 2:
                    # Adding t1wce
                    t1wce.append(cv2.resize(image, (input_height, input_width)))
                elif m == 3:
                    # Adding t2w
                    t2w.append(cv2.resize(image,   (input_height, input_width)))
            
        if modeling_in == '3D':
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
                
            return np.array((
                flair, t1w,
                t1wce, t2w),
                dtype="object").T, self.label.iloc[index,]
    
# data loader 
class BrainTSGeneratorRegistered(tf.keras.utils.Sequence):
    def __init__(self, dicom_path, data):
        self.data = data
        self.dicom_path = dicom_path
        self.label = self.data['MGMT_value']
         
    def __len__(self):
        return len(self.data['BraTS21ID'])
    
    def __getitem__(self, index):
        patient_ids = f"{str(self.data['BraTS21ID'][index]).zfill(5)}/"
   
        # for 3D modeling 
        flair = []
        t1w = []
        t1wce = []
        t2w = [] 

        
        # Iterating over each modality
        for m, t in enumerate(input_modality):
            voxel = load_voxel(patient_ids, t, "train", input_height)
            voxel = voxel[len(voxel) // 2: (len(voxel) // 2) + input_depth]
            voxel = read_dicom_xray_and_normalize(path=None, voxel=voxel)
        
            if modeling_in == '3D':
                if m == 0:
                    # Adding flair 
                    flair.append(voxel)
                elif m == 1:
                    # Adding t1w
                    t1w.append(voxel)
                elif m == 2:
                    # Adding t1wce
                    t1wce.append(voxel)
                elif m == 3:
                    # Adding t2w
                    t2w.append(voxel)
                    

        if modeling_in == '3D':
            X = np.stack( 
                (
                    np.moveaxis(flair[0], 0, -1),
                    np.moveaxis(t1w[0], 0, -1),
                    np.moveaxis(t1wce[0], 0, -1),
                    np.moveaxis(t2w[0], 0, -1)
                ), 
                axis=-1
            )
            return X, self.label.iloc[index,]
    
    
    
    
# Function to read dicom file 
def read_dicom_xray_and_normalize(path=None, voxel=None):
    if voxel is None:
        dicom = pydicom.read_file(path)
        data  = dicom.pixel_array
    else:
        data = voxel

    if data.mean() <= 0.8:
        # If all black, return data and find non-black if possible.
        return data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def load_voxel(study_id, scan_type="FLAIR", split="train", sz=256):
    data_root = Path('D:/Kaggle & DataSets/Kaggle/BrainTumor/registered_brain_tumor_kaggle')
    npy_path = Path(data_root).joinpath(split, study_id, f"{scan_type}.npy")
    voxel = np.load(str(npy_path))
    return voxel
