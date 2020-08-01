# -*- coding: utf-8 -*-
""" @author: Sumit Anand """

#%% Importing Libraries

import numpy as np
import os
import time
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.layers import Dense, Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

#%% Preparing Input test image to feed CovNet neural network

img_path = 'TestImage/testImage.jpg'
img = image.load_img(img_path, target_size=(224, 224))
#Image is nothing but data of array so convert the image pixels to a numpy array
testImageData = image.img_to_array(img)
print('Input image shape before expanding dimensions:', testImageData.shape)
#Reshape image data for the model
testImageData = np.expand_dims(testImageData, axis=0)
# prepare the image for the VGG model
testImageData = preprocess_input(testImageData)
print('Input image shape:', testImageData.shape)

#%% Preprocessing of training data

# Get the current working directory
CurrentDirectory = os.getcwd()
# Define train data path
train_data_path = CurrentDirectory + '/data'
data_dir_list = os.listdir(train_data_path)

# List of all preprocessed image data 
img_data_list = []

for datasetFolder in data_dir_list:
    print ('Loaded folders of images dataset - {}' .format(datasetFolder))
    #List all images in a folder
    img_list = os.listdir(train_data_path + '/' + datasetFolder)
    print ('List of all images - '+'{}\n'.format(img_list))
    #loop over each image and preprocess the image for the model
    for img in img_list:
        img_path = train_data_path + '/'+ datasetFolder + '/'+ img        
        print('image : {} ' .format(img))
        img = image.load_img(img_path, target_size=(224, 224))
        eachImageData = image.img_to_array(img)
        eachImageData = np.expand_dims(eachImageData, axis=0)
        eachImageData = preprocess_input(eachImageData)
        print('image shape : {}' .format(eachImageData.shape))
        img_data_list.append(eachImageData)

# Preprocess the image list
img_data = np.array(img_data_list)
print ('Image array data {} ' .format(img_data.shape))
img_data=np.rollaxis(img_data,1,0)
print ('Image array data after roll axis {} ' .format(img_data.shape))
img_data=img_data[0]
print ('Image array data after eliminating first param {} ' .format(img_data.shape))


# Define the number of classes
num_classes = 4
# Total number of trainable image samples
print ('Image array data {} ' .format(img_data.shape[0]))
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')
print ('Labels before assigning {} ' .format(labels))
# Separate Label based on your trainable categorial image count
labels[0:202]=0
labels[202:404]=1
labels[404:606]=2
labels[606:]=3
print ('Labels after assigning {} ' .format(labels))

names2 = ['cats','dogs','horses','humans']

# Convert class labels to on-hot encoding ( binary class matrix) for matching against each labels
binaryClassMatrix = np_utils.to_categorical(labels)

# Shuffle the dataset
x,y = shuffle(img_data, binaryClassMatrix, random_state=2)
# Split the dataset and store in variable
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=2)


#%% Train & save Custom VGG model

#Training the classifier alone
image_input = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()
# Get the last layer from CovNet layer 
last_layer = model.get_layer('block5_pool').output
print(' Layer got from Model {}' .format(last_layer))
x= Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
# Add a custom layer with 4 classes where input is the output of the  second last layer 
custom_layer = Dense(num_classes, activation='softmax', name='output')(x)
# Create custom model from image_input data & custom_layer
custom_vgg_model = Model(image_input, custom_layer)
custom_vgg_model.summary()

# Freeze all non required layer
for layer in custom_vgg_model.layers[:-1]:
	layer.trainable = False

# Check layers are trainable or not
custom_vgg_model.layers[3].trainable
# Compile the custom model
custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

model_path = "Models/VGG16-FineTuning.h5"
checkpoint_dir = os.path.dirname(model_path)

# Create a callback that saves the model's weights after every epochs
cp_callback = ModelCheckpoint(filepath=model_path, save_weights_only=True, verbose=1)

t=time.time()
# Train the model
hist = custom_vgg_model.fit(X_train, 
                            Y_train, 
                            batch_size=32, 
                            epochs=12, 
                            verbose=1, 
                            validation_data=(X_test, Y_test), 
                            callbacks=[cp_callback])

print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model.evaluate(X_test, Y_test, batch_size=10, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
# Save the model
custom_vgg_model.save(model_path)

#%% Benchmarking & plotting model accuracy 


# visualizing losses and accuracy
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
xc = range(12)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])
plt.show()

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])
plt.show()

#%% Test with images


