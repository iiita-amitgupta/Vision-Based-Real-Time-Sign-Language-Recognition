



# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


##
import sklearn                                        #importing sklearn , it is buit up of numpy, pandas, and matplotlib
from sklearn import metrics                           #importing metrics for the classification
import sklearn.metrics as metrics                     #importing sklearn metrics for showing classification result
from keras.optimizers import Adam                     #importing Adam optimizer 
#import numpy                                          #imporing numpy
from matplotlib import pyplot as plt                  #impoting matplotlib to show model accuracy graph and model loss graph 
#from keras import history
##

# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Dropout(0.25))

# third convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

##
#adding dropout layer to remove 25% neuron
classifier.add(Dropout(0.25))  
##

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))

##
#adding dropout layer to remove 25% neuron
classifier.add(Dropout(0.25)) 

#adding fully-connected layer 2 
classifier.add(Dense(units=128, activation='relu'))
##

classifier.add(Dense(units=6, activation='softmax')) 

# Compiling the CNN
classifier.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2



# Step 2 - Preparing the train/test data and training the model

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        channel_shift_range=0.2,)

test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=5,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=5,
                                            color_mode='grayscale',
                                            class_mode='categorical') 
history = classifier.fit_generator(
        training_set,
        steps_per_epoch=900, # No of images in training set
        epochs=4,
        validation_data=test_set,
        validation_steps=90)# No of images in test set


# Saving the model
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model-bw.h5')










