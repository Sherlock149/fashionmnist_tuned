# -*- coding: utf-8 -*-
"""
Created on Wed May 26 17:55:48 2021

@author: abhishek
"""

from tensorflow import keras
from tensorflow.keras.models import load_model
from keras_preprocessing.image import img_to_array,load_img
import numpy as np

model = load_model('cnn2/tuned_cnn2_model.h5')

# the mnest labels
labels = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']


import cv2
# convert to greyscale and resize,reshape,preprocess
img = cv2.imread('cnn2/test_images/img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = cv2.resize(img, (28,28))
img = img/255.0
img = img.reshape((1,)+img.shape+(1,))


res = model.predict(img)
i = np.argmax(res)

print(labels[i])



# Uncomment to get accuracy score with test dataset
# Comment the above part
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from keras_preprocessing.image import img_to_array,load_img
import numpy as np

# importing the fashion_mnist dataset
# already available in keras
# It contains 10000 28x28 images
fashion_mnist = keras.datasets.fashion_mnist

#train and test data
# Here X is images Y is labels
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

test_images = test_images/255.0


#convert to 4D to pass into numpy
test_images = test_images.reshape(test_images.shape+(1,))

classifier = load_model('cnn2/tuned_cnn2_model.h5')
y_predict = classifier.predict(test_images)
res = np.zeros(10000)
ptr = 0
for i in y_predict:
    res[ptr] = np.argmax(y_predict[ptr])
    ptr+=1

from sklearn.metrics import accuracy_score

score = accuracy_score(test_labels, res)

print("Test Accuracy:",score*100,"%")

"""