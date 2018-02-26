### Copyright (C) Microsoft Corporation.  

from keras.layers import Dense
from keras.models import Model
from keras_contrib.applications.densenet import DenseNetImageNet121
import keras_contrib

def build_model(crt_densenet_function):
    """

    Returns: a model with specified weights

    """
    # define the model, use pre-trained weights for image_net
    base_model = crt_densenet_function(input_shape=(224, 224, 3),
                                     weights='imagenet',
                                     include_top=False,
                                     pooling='avg')

    x = base_model.output
    predictions = Dense(14, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

if __name__=="__main__":        
    model = build_model(DenseNetImageNet121)
    print(model.summary())    
    model = build_model(keras_contrib.applications.densenet.DenseNetImageNet201)
    print(model.summary())