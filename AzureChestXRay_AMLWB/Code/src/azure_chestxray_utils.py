### Copyright (C) Microsoft Corporation.  

import os
import numpy as np
import pandas as pd 

class chestxray_consts(object):
    DISEASE_list = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax',
                'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']
    
    PRETRAINED_DENSENET201_IMAGENET_CHESTXRAY_MODEL_FILE_NAME =  'chexnet_14_weights_multigpu_contribmodel_121layer_712split_epoch_011_val_loss_153.9783.hdf5'
    FULLY_PRETRAINED_MODEL_DIR_list = [ 'fully_trained_models']


    CHESTXRAY_MODEL_EXPECTED_IMAGE_HEIGHT  = 224
    CHESTXRAY_MODEL_EXPECTED_IMAGE_WIDTH = 224

    BASE_INPUT_DIR_list = ['chestxray', 'data', 'ChestX-ray8']
    BASE_OUTPUT_DIR_list = ['chestxray', 'output']
    CREDENTIALS_DIR_list = ['code', 'notShared']

    SRC_DIR_list = ['Code',  'src']
    ChestXray_IMAGES_DIR_list = ['ChestXray-NIHCC']
    ChestXray_OTHER_DATA_DIR_list = ['ChestXray-NIHCC_other']
    PROCESSED_IMAGES_DIR_list = ['processed_npy14']
    DATA_PARTITIONS_DIR_list = ['data_partitions']
    MODEL_WEIGHTS_DIR_list = [ 'weights_tmpdir']

    def __setattr__(self, *_):
        raise TypeError


# os agnostic 'ls' function
def get_files_in_dir(crt_dir):
        return( [f for f in os.listdir(crt_dir) if os.path.isfile(os.path.join(crt_dir, f))])
        
       
    
def normalize_nd_array(crt_array):
    # Normalised [0,1]
    crt_array = crt_array - np.min(crt_array)
    return(crt_array/np.ptp(crt_array))

def print_image_stats_by_channel(crt_image):
    print('min:')
    print(np.amin(crt_image[:,:,0]), 
          np.amin(crt_image[:,:,1]),
          np.amin(crt_image[:,:,2]))
    print('max:')
    print(np.amax(crt_image[:,:,0]), 
          np.amax(crt_image[:,:,1]),
          np.amax(crt_image[:,:,2]))        

class bbox_NIH_data():
    def __init__(self, bbox_data_file_dir, bbox_data_file = 'BBox_List_2017.csv'):
        all_bbox_data = pd.read_csv(os.path.join(bbox_data_file_dir, bbox_data_file))

        # show some stats
        # for tallying, collections lib is faster than list comprehension
        from collections import Counter
        pathologies_distribution = Counter(list(all_bbox_data['Finding Label']))
        pathologies_distribution = sorted(pathologies_distribution.items(), key=lambda x: x[1], reverse=True)

        print('Pathologies distribution:')
        print(pathologies_distribution)
        
        self.all_bbox_data = all_bbox_data
         
        print("Loaded {} bbox records".format(self.all_bbox_data.shape))
    
    def get_patologies_images(self, crt_pathology_name_list):
        
        #  more complex code needed if bbox data has multiple labels per record 
        # something like (intersect = set.intersection(*crt_pathology_name_list)) per row
        return self.all_bbox_data[self.all_bbox_data['Finding Label'].isin(crt_pathology_name_list)][['Image Index', 'Finding Label']]

        
        
if __name__=="__main__":        
    prj_consts = chestxray_consts()
    print('model_expected_image_height = ', prj_consts.CHESTXRAY_MODEL_EXPECTED_IMAGE_HEIGHT)
    print('model_expected_image_width = ', prj_consts.CHESTXRAY_MODEL_EXPECTED_IMAGE_WIDTH)
    
    # crt_bbox_data = bbox_NIH_data(other_data_dir, 'BBox_List_2017.csv')
    # crt_pathology_image_file_names = crt_bbox_data.get_patologies_images(list([ 'Nodule'])) # ['Cardiomegaly', 'Infiltrate']
    # print(crt_pathology_image_file_names[:5])
    # print(crt_pathology_image_file_names.shape)
