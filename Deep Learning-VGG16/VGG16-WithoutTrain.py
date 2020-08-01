# -*- coding: utf-8 -*-
""" @author: Sumit Anand """

#%% Importing Libraries

import numpy as np
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions


#%% VGG-16 Pre-trained model Prediction with topLayer

#param include_top (True): Boolean flag represents include the 3 fully-connected layers at the top of the network. You don’t need these if you are fitting the model on your own problem.
#param weights ('imagenet'): What weights to load. You can specify None to not load pre-trained weights if you are interested in training the model yourself from scratch.
#Downloaded loaction of model : (C:\Users\USERNAME\.keras\models)
model = VGG16(include_top=True, weights='imagenet')
model.summary()

#Input test image with dimensions-1600x1200
img_path = 'TestImage/testImage.jpg'
img = image.load_img(img_path, target_size=(224, 224))
#Image is nothing but data of array so convert the image pixels to a numpy array
imageData = image.img_to_array(img)
print('Input image shape before expanding dimensions:', imageData.shape)
#Reshape data for the model
imageData = np.expand_dims(imageData, axis=0)
# prepare the image for the VGG model
imageData = preprocess_input(imageData)
print('Input image shape:', imageData.shape)

# predict the probability across all output classes
prediction_with_topLayer = model.predict(imageData)
#Decode predictions using ImageNet predefined classes as predictedLabel
predictedLabel = decode_predictions(prediction_with_topLayer)

# retrieve the most likely result, e.g. highest probability
predictedLabel = predictedLabel[0][0]

print('Predicted Image : %s (%.2f%%)' % (predictedLabel[1], predictedLabel[2]*100)) 


#%% VGG-16 Pre-trained model Prediction without topLayer

model = VGG16(include_top=False, weights='imagenet' )
model.summary()

#Input test image with dimensions-1600x1200
img_path = 'TestImage/testImage.jpg'
img = image.load_img(img_path, target_size=(224, 224))
#Image is nothing but data of array so convert the image pixels to a numpy array
imageData = image.img_to_array(img)
print('Input image shape before expanding dimensions:', imageData.shape)
#Reshape data for the model
imageData = np.expand_dims(imageData, axis=0)
# prepare the image for the VGG model
imageData = preprocess_input(imageData)
print('Input image shape:', imageData.shape)

# predict the probability across all output classes
prediction_WITHOUT_topLayer = model.predict(imageData)

#Decode predictions using ImageNet predefined classes as predictedLabel
predictedLabel = decode_predictions(prediction_WITHOUT_topLayer)

# retrieve the most likely result, e.g. highest probability
predictedLabel = predictedLabel[0][0]

print('Predicted Image : %s (%.2f%%)' % (predictedLabel[1], predictedLabel[2]*100)
      
   
