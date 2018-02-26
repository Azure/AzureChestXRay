# This script generates the scoring and schema files
# Creates the schema, and holds the init and run functions needed to 
# operationalize the chestXray model


import os, sys, pickle, base64
import keras.models
import keras.layers
import keras_contrib.applications.densenet
import pandas as pd
import numpy as np
import azure_chestxray_utils, azure_chestxray_cam

####################################
# Parameters
####################################
global chest_XRay_model
global as_string_b64encoded_pickled_data_column_name
as_string_b64encoded_pickled_data_column_name   = 'encoded_image'
global densenet_weights_file_name
# densenet_weights_file_name = 'weights_only_chestxray_model_14_weights_712split_epoch_029_val_loss_147.7599.hdf5'
densenet_weights_file_name = 'weights_only_chestxray_model_14_weights_712split_epoch_029_val_loss_147.7599 - Copy.hdf5'

# Import data collection library. Only supported for docker mode.
# Functionality will be ignored when package isn't found
try:
    from azureml.datacollector import ModelDataCollector
except ImportError:
    print("Data collection is currently only supported in docker mode. May be disabled for local mode.")
    # Mocking out model data collector functionality
    class ModelDataCollector(object):
        def nop(*args, **kw): pass
        def __getattr__(self, _): return self.nop
        def __init__(self, *args, **kw): return None
    pass

####################################
# Utils
####################################
def as_string_b64encoded_pickled(input_object):
     #b64encode returns bytes class, make it string by calling .decode('utf-8')
     return (base64.b64encode(pickle.dumps(input_object))).decode('utf-8')

def unpickled_b64decoded_as_bytes(input_object):
    if input_object.startswith('b\''):
        input_object = input_object[2:-1]
    # make string bytes
    input_object   =  input_object.encode('utf-8')
    #decode and the unpickle the bytes to recover original object
    return (pickle.loads(base64.b64decode(input_object)))

def get_image_score_and_serialized_cam(crt_cv2_image, crt_chest_XRay_model):
    prj_consts = azure_chestxray_utils.chestxray_consts()
    crt_cv2_image = azure_chestxray_utils.normalize_nd_array(crt_cv2_image)
    crt_cv2_image = 255*crt_cv2_image
    crt_cv2_image=crt_cv2_image.astype('uint8')
    predictions, cam_image, predicted_disease_index = \
    azure_chestxray_cam.get_score_and_cam_picture(crt_cv2_image, crt_chest_XRay_model)
    blended_image = azure_chestxray_cam.process_cam_image(cam_image, crt_cv2_image)
    serialized_image = azure_chestxray_cam.plot_cam_results(blended_image, cam_image, crt_cv2_image, \
                 prj_consts.DISEASE_list[predicted_disease_index])
    return predictions, serialized_image

####################################
# API functions
####################################

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.
def init():
    try: 
        print("init() method: Python version: " + str(sys.version))
        print("crt Dir: " + os.getcwd())
        
        import pip
        # pip.get_installed_distributions()
        myDistr = pip.get_installed_distributions()
        type(myDistr)
        for crtDist in myDistr:
            print(crtDist)

        # load the model file
        global chest_XRay_model
        chest_XRay_model = azure_chestxray_utils.build_DenseNetImageNet201_model() 
        chest_XRay_model.load_weights(densenet_weights_file_name)
        print('Densenet model loaded')
        
    except Exception as e:
        print("Exception in init:")
        print(str(e))

def run(input_df):
    try:
        import json
        
        debugCounter = 0
        print("run() method: Python version: " + str(sys.version) ); print('Step '+str(debugCounter));debugCounter+=1

        print ('\ninput_df shape {}'.format(input_df.shape))
        print(list(input_df))
        print(input_df)

        input_df = input_df[as_string_b64encoded_pickled_data_column_name][0]; print('Step '+str(debugCounter));debugCounter+=1
        input_cv2_image = unpickled_b64decoded_as_bytes(input_df); print('Step '+str(debugCounter));debugCounter+=1

        #finally scoring
        predictions, serialized_cam_image = get_image_score_and_serialized_cam(input_cv2_image, chest_XRay_model)
        #predictions = chest_XRay_model.predict(input_cv2_image[None,:,:,:])

        # prediction_dc.collect(ADScores)
        outDict = {"chestXrayScore": str(predictions), "chestXrayCAM":as_string_b64encoded_pickled(serialized_cam_image)}
        return json.dumps(outDict)
    except Exception as e:
        return(str(e))


####################################
# main function can be used for test and demo
####################################
def main():
    from azureml.api.schema.dataTypes import DataTypes
    from azureml.api.schema.sampleDefinition import SampleDefinition
    from azureml.api.realtime.services import generate_schema

    print('Entered main function:')
    print(os.getcwd())
    
    amlWBSharedDir = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'] 
    print(amlWBSharedDir)

    def get_files_in_dir(crt_dir):
        return( [f for f in os.listdir(crt_dir) if os.path.isfile(os.path.join(crt_dir, f))])

    fully_trained_weights_dir=os.path.join(
        amlWBSharedDir,
        os.path.join(*(['chestxray', 'output',  'trained_models_weights'])))
    crt_models = get_files_in_dir(fully_trained_weights_dir)
    print(fully_trained_weights_dir)
    print(crt_models)

    test_images_dir=os.path.join(
        amlWBSharedDir, 
        os.path.join(*(['chestxray', 'data', 'ChestX-ray8', 'test_images'])))
    test_images = get_files_in_dir(test_images_dir)
    print(test_images_dir)
    print(len(test_images))

    # score in local mode (i.e. here in main function)
    model = azure_chestxray_utils.build_DenseNetImageNet201_model()
    model.load_weights(os.path.join(
        fully_trained_weights_dir, densenet_weights_file_name))

    print('Model weoghts loaded!')

    import cv2
    cv2_image = cv2.imread(os.path.join(test_images_dir,test_images[0]))
    x, serialized_cam_image = get_image_score_and_serialized_cam(cv2_image, model)
    file_bytes = np.asarray(bytearray(serialized_cam_image.read()), dtype=np.uint8)
    recovered_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # x = model.predict(cv2_image[None,:,:,:])
    print(test_images[0])
    print(x)
    print(recovered_image.shape)

    #  score in local mode (i.e. here in main function) using encoded data
    encoded_image = as_string_b64encoded_pickled(cv2_image)
    df_for_api = pd.DataFrame(data=[[encoded_image]], columns=[as_string_b64encoded_pickled_data_column_name])
    del encoded_image 
    del cv2_image
    del serialized_cam_image
    
    input_df = df_for_api[as_string_b64encoded_pickled_data_column_name][0]
    input_cv2_image = unpickled_b64decoded_as_bytes(input_df); 
    x, serialized_cam_image = get_image_score_and_serialized_cam(input_cv2_image, model) 
    file_bytes = np.asarray(bytearray(serialized_cam_image.read()), dtype=np.uint8)
    recovered_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # x = model.predict(input_cv2_image[None,:,:,:])
    print('After encoding and decoding:')
    print(x)
    print(recovered_image.shape)

    del model

    # now create the post deployment env, i.e. score using init() and run()
    crt_dir = os.getcwd()
    working_dir = os.path.join(crt_dir, 'tmp_cam_deploy')
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    import shutil
    shutil.copyfile(
        os.path.join( fully_trained_weights_dir,densenet_weights_file_name), 
        os.path.join( working_dir,densenet_weights_file_name)) 

    os.chdir(working_dir)

    # Turn on data collection debug mode to view output in stdout
    os.environ["AML_MODEL_DC_DEBUG"] = 'true'

    # Test the output of the functions
    init()
    print("Result: " + run(df_for_api))

     # #Generate the schema
    data_for_schema = {"input_df": SampleDefinition(DataTypes.PANDAS, df_for_api)}
    schema_file = os.path.join(fully_trained_weights_dir, 'chest_XRay_cam_service_schema.json')
    generate_schema(run_func=run, inputs=data_for_schema, filepath=schema_file)
    print("Schema saved in " +schema_file)   
    

if __name__ == "__main__":
    main()
