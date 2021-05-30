# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:17:16 2021

@author: abhishek
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np

# importing the fashion_mnist dataset
# already available in keras
# It contains 10000 28x28 images
fashion_mnist = keras.datasets.fashion_mnist

#train and test data
# Here X is images Y is labels
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

#feature scaling from 0-255 to 0-1
train_images = train_images/255.0
test_images = test_images/255.0

#convert to 4D to pass into numpy
# 1 is passed as a tuple as (1,)
# .reshape(no. of images, pixelxpixel, RGB or Grey) ->This is the fixed syntax
train_images = train_images.reshape(train_images.shape+(1,))
test_images = test_images.reshape(test_images.shape+(1,))

# Hyperparameter tuning
from tensorflow.keras import layers
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

""" This time use Conv2D followed by flatten for hidden layer
    Last layer remains similar
    num_layers: Number of Conv2D layers needed
    units: Number of filters in each layer
    kernal: Size of filters
"""
"""
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(layers.Conv2D(filters=hp.Int('units_' + str(i),min_value=32,max_value=80,step=16),
                                kernel_size = hp.Choice('kernal'+str(i),[3,5]), 
                                activation='relu'))
    
    # Flatten Layer
    model.add(layers.Flatten())
    
    # 1 Dense Layer, number of neurons is decided
    model.add(layers.Dense(units=hp.Int('dense_units',min_value=32,
                                        max_value=128,step=16), activation='relu'))
        
    # last layer: Chage acc to dataset
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model
"""
def build_model(hp):
    model = keras.Sequential([
        layers.Conv2D(filters=hp.Int('conv_1_filter',min_value=32,max_value=128,step=16),
                                kernel_size = hp.Choice('kernal 1',[3,5]), 
                                activation='relu',input_shape=(28,28,1)),
        
        layers.Conv2D(filters=hp.Int('conv_2_filter',min_value=32,max_value=64,step=16),
                                kernel_size = hp.Choice('kernal 2',[3,5]), 
                                activation='relu'),
        
        # Flatten Layer
        layers.Flatten(),
    
        # 1 Dense Layer, number of neurons is decided
        layers.Dense(units=hp.Int('dense_units',min_value=32,
                                        max_value=128,step=16), activation='relu'),
        
        # last layer: Chage acc to dataset
        layers.Dense(10, activation='softmax')
        ])
    
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
# Running randomSearch acc to the above model and saving the results in tuner

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='cnn2',
    project_name='tuner_summary')


tuner.search_space_summary()

tuner.search(train_images, train_labels, epochs=3, validation_split=0.1)
tuner.results_summary()

#taking the best model for training
classifier = tuner.get_best_models(num_models=1)[0]
classifier.summary()

# using this parameters for training and saving the model

classifier.fit(train_images,train_labels,epochs=10,validation_split=0.1)
classifier.save('cnn2/tuned_cnn2_model.h5')