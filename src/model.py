from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import math
import os
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from preprocess import x_train, x_val, y_train, y_val


def eval_lstm(individual):

    #func_seq = 2
    IND_SIZE = 8

    n_steps=10
    n_features =1

    num_params = 2
    input_layer_flag = False
    return_flag = False

    model = Sequential() 

    for i in range(IND_SIZE):
        
        index = i*num_params
        
        if individual[index] > 0:
            if input_layer_flag==False:
                model.add(LSTM(individual[index+1],activation='relu',
                            input_shape=(n_steps, n_features),
                                return_sequences=True))
        
                input_layer_flag=True
        
            else:
                model.add(LSTM(individual[index+1],activation='relu', 
                            return_sequences=True))
        
            return_flag=True
    
    # final layer
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    hist = model.fit(x_train, y_train, batch_size = 32, 
                     validation_data= (x_val, y_val) ,
                     epochs=2 ,verbose=2)
    
    print(hist.history['val_loss'])
    return hist.history['val_loss'][1], 


def eval_cnn(individual):

    
    
    num_conv_params = len(func_seq)
    input_layer_flag = False
    return_flag = False
    dim = 0
    index = 0

    model = Sequential() 

    for i in range(max_conv_layer):
        
        index = i*num_conv_params
        
        if individual[index] > 0:
            if input_layer_flag==False:
                model.add(Conv1D(individual[index+1],3,
                        padding='same',
                            input_shape=(12, 1)))
        
                input_layer_flag=True
        
            else:
                model.add(Conv1D(individual[index+1],3,
                        padding='same'))
        
            return_flag=True

            if individual[index + 2]:
                    model.add(BatchNormalization())
            model.add(Activation(individual[index + 3]))
            model.add(Dropout(float(individual[index + 4] / 20.0)))
            max_pooling_type = individual[index + 5]
            
            # must be large enough for a convolution
            if max_pooling_type == 1 and dim >= 5:
                model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
                dim = int(math.ceil(dim / 2))
    
    # final layer
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    

    hist = model.fit(x_train, y_train, batch_size = 16, validation_data = (x_val, y_val), epochs=2 ,verbose=2)
    
    return hist.history["val_loss"][1],