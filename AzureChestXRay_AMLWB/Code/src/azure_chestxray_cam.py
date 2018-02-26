### Copyright (C) Microsoft Corporation.  

import keras.backend as K
import sys, os, io
import numpy as np
import cv2

import matplotlib
matplotlib.use('agg')

paths_to_append = [os.path.join(os.getcwd(), os.path.join(*(['Code',  'src'])))]
def add_path_to_sys_path(path_to_append):
    if not (any(path_to_append in paths for paths in sys.path)):
        sys.path.append(path_to_append)
[add_path_to_sys_path(crt_path) for crt_path in paths_to_append]

import azure_chestxray_utils


def get_score_and_cam_picture(cv2_input_image, DenseNetImageNet121_model):
# based on https://github.com/jacobgil/keras-cam/blob/master/cam.py
    width, height, _ = cv2_input_image.shape
    class_weights = DenseNetImageNet121_model.layers[-1].get_weights()[0]
    final_conv_layer = DenseNetImageNet121_model.layers[-3]
    get_output = K.function([DenseNetImageNet121_model.layers[0].input], 
                            [final_conv_layer.output, \
                             DenseNetImageNet121_model.layers[-1].output])
    [conv_outputs, prediction] = get_output([cv2_input_image[None,:,:,:]])
    conv_outputs = conv_outputs[0, :, :, :]
    prediction = prediction[0,:]
    
    #Create the class activation map.
    predicted_disease = np.argmax(prediction)
    cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[:2])
    for i, w in enumerate(class_weights[:, predicted_disease]):
            cam += w * conv_outputs[:, :, i]
    
    return prediction, cam, predicted_disease


def process_cam_image(crt_cam_image, xray_image, crt_alpha = .5):
    im_width, im_height, _ = xray_image.shape
    crt_cam_image = cv2.resize(crt_cam_image, (im_width, im_height), \
                               interpolation=cv2.INTER_CUBIC)
    
#     do some gamma enhancement, e is too much
    crt_cam_image = np.power(1.1, crt_cam_image)
    crt_cam_image = azure_chestxray_utils.normalize_nd_array(crt_cam_image)
    # crt_cam_image[np.where(crt_cam_image < 0.5)] = 0 
    crt_cam_image = 255*crt_cam_image

    # make cam an rgb image
    empty_image_channel = np.zeros(dtype = np.float32, shape = crt_cam_image.shape[:2])
    crt_cam_image = cv2.merge((crt_cam_image,empty_image_channel,empty_image_channel))
    
    blended_image = cv2.addWeighted(xray_image.astype('uint8'),crt_alpha,\
                                    crt_cam_image.astype('uint8'),(1-crt_alpha),0)
    return(blended_image)

def plot_cam_results(crt_blended_image, crt_cam_image, crt_xray_image, map_caption):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize = (15,7))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(crt_xray_image, cmap = 'gray', interpolation = 'bicubic')
    ax1.set_title('Orig X Ray')
    plt.axis('off')

    ax2 = fig.add_subplot(2,3, 2)
    cam_plot = ax2.imshow(crt_cam_image, cmap=plt.get_cmap('OrRd'), interpolation = 'bicubic')
    plt.colorbar(cam_plot, ax=ax2)
    ax2.set_title('Activation Map')
    plt.axis('off')

    ax3 = fig.add_subplot(2,3, 3)
    blended_plot = ax3.imshow(crt_blended_image, interpolation = 'bicubic')
    plt.colorbar(cam_plot, ax=ax3)
    ax3.set_title(map_caption)
    plt.axis('off')
    
    # serialize blended image plot padded in the x/y-direction
    image_as_BytesIO = io.BytesIO()
    x_direction_pad = 1.05;y_direction_pad=1.2
    extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(image_as_BytesIO, 
                bbox_inches=extent.expanded(x_direction_pad, 
                                            y_direction_pad),
               format='png')
    image_as_BytesIO.seek(0)
    return(image_as_BytesIO)
    

    
def process_xray_image(crt_xray_image, DenseNetImageNet121_model):

#     print(crt_xray_image.shape)
    crt_xray_image = azure_chestxray_utils.normalize_nd_array(crt_xray_image)
    crt_xray_image = 255*crt_xray_image
    crt_xray_image=crt_xray_image.astype('uint8')

    crt_predictions, crt_cam_image, predicted_disease_index = \
    get_score_and_cam_picture(crt_xray_image, 
                              DenseNetImageNet121_model)
    
    prj_consts = azure_chestxray_utils.chestxray_consts()
    likely_disease=prj_consts.DISEASE_list[predicted_disease_index]
    likely_disease_prob = 100*crt_predictions[predicted_disease_index]
    likely_disease_prob_ratio=100*crt_predictions[predicted_disease_index]/sum(crt_predictions)
    print('predictions: ', crt_predictions)
    print('likely disease: ', likely_disease)
    print('likely disease prob: ', likely_disease_prob)
    print('likely disease prob ratio: ', likely_disease_prob_ratio)
    
    crt_blended_image = process_cam_image(crt_cam_image, crt_xray_image)
    plot_cam_results(crt_blended_image, crt_cam_image, crt_xray_image,
                    str(likely_disease)+ ' ' +
                    "{0:.1f}".format(likely_disease_prob)+ '% (weight ' +
                    "{0:.1f}".format(likely_disease_prob_ratio)+ '%)')

def process_nih_data(nih_data_files, NIH_data_dir, DenseNetImageNet121_model):
    for crt_image in nih_data_files:
        # print(crt_image)
        prj_consts = azure_chestxray_utils.chestxray_consts()

        crt_xray_image = cv2.imread(os.path.join(NIH_data_dir,crt_image))
        crt_xray_image = cv2.resize(crt_xray_image, 
                                    (prj_consts.CHESTXRAY_MODEL_EXPECTED_IMAGE_HEIGHT, 
                                     prj_consts.CHESTXRAY_MODEL_EXPECTED_IMAGE_WIDTH)) \
                        .astype(np.float32)

        process_xray_image(crt_xray_image, DenseNetImageNet121_model )   
        
if __name__=="__main__":
    #FIXME
    # add example/test code here



    NIH_annotated_Cardiomegaly = ['00005066_030.png']
    data_dir = ''
    cv2_image = cv2.imread(os.path.join(data_dir,NIH_annotated_Cardiomegaly[0]))

    print_image_stats_by_channel(cv2_image)
    cv2_image = normalize_nd_array(cv2_image)
    cv2_image = 255*cv2_image
    cv2_image=cv2_image.astype('uint8')
    print_image_stats_by_channel(cv2_image)

    predictions, cam_image, predicted_disease_index = get_score_and_cam_picture(cv2_image, model)
    print(predictions)
    prj_consts = azure_chestxray_utils.chestxray_consts()
    print(prj_consts.DISEASE_list[predicted_disease_index])
    print('likely disease: ', prj_consts.DISEASE_list[predicted_disease_index])
    print('likely disease prob ratio: ', \
          predictions[predicted_disease_index]/sum(predictions))
    blended_image = process_cam_image(cam_image, cv2_image)
    plot_cam_results(blended_image, cam_image, cv2_image, \
                 prj_consts.DISEASE_list[predicted_disease_index])      