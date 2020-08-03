
#importing keras library
import keras                                          
from keras.models import Sequential                   
from keras.layers import Convolution2D                
from keras.layers import MaxPooling2D                 
from keras.layers import Flatten                      
from keras.layers import Dense, Dropout               
import sklearn                                        
from sklearn import metrics                           
import sklearn.metrics as metrics                     
from keras.optimizers import Adam                      
#import numpy                                          
from matplotlib import pyplot as plt                   

classifier = Sequential()                             #creating CNN classifier object


#adding Convolution layer 1
classifier.add(Convolution2D(32, (3, 3), input_shape=(200, 200, 3), activation='relu')) 
classifier.add(MaxPooling2D(pool_size=(2, 2)))                                        

#adding convolution layer 2
classifier.add(Convolution2D(32, (2, 2), activation='relu'))                          
classifier.add(MaxPooling2D(pool_size=(2, 2)))                                        

#adding Convolution layer 3
classifier.add(Convolution2D(32, (2, 2), activation='relu'))                          
classifier.add(MaxPooling2D(pool_size=(2, 2)))                                        

#adding Convolution layer 4
classifier.add(Convolution2D(32, (2, 2), activation='relu'))                          
classifier.add(MaxPooling2D(pool_size=(2, 2)))                                        

#adding dropout layer to remove 25% neuron
classifier.add(Dropout(0.30))                                                         

#adding flatten layer
classifier.add(Flatten())                                                             

#adding fully-connected layer 1
classifier.add(Dense(units=256, activation='relu'))                                   
#adding dropout layer to remove 25% neuron
classifier.add(Dropout(0.25))                                                         
#adding fully-connected layer 2 
classifier.add(Dense(units=256, activation='relu'))                                   

#adding output layer to classify image
classifier.add(Dense(units=29, activation='softmax'))                                

#compiling the CNN
classifier.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) 
#model are updated each time ,how much to change the model in response to the eastimated error



#import ImageDataGenerator class 
from keras.preprocessing.image import ImageDataGenerator                               

#making object to augment training dataset
train_datagen = ImageDataGenerator(                                                   
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
		channel_shift_range=0.2,
		    width_shift_range=0.2,
			    height_shift_range=0.2,)

test_datagen = ImageDataGenerator(rescale=1./255)                                    #making object to testing dataset

training_set = train_datagen.flow_from_directory('data/train',                       
                                                 target_size=(200, 200),               
                                                 batch_size=10,                      #batch size to pick image at a time
                                                 color_mode='rgb',             
                                                 class_mode='categorical')           #categorical is used for image class mode

test_set = test_datagen.flow_from_directory('data/test',                             
                                            target_size=(200, 200),                  
                                            batch_size=10,                           
                                            color_mode='rgb',                  
                                            class_mode='categorical')                #categorical is used for image class mode
history=classifier.fit_generator(                                                    
        training_set,                                                                
        steps_per_epoch=2900,                                                        
        epochs=25,                                                                   
        validation_data=test_set,                                                    
        validation_steps=1160)                                                       

   

#saving the model
model_json = classifier.to_json()                                                    
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model-bw.h5')



plt.plot(history.history['accuracy'])                                                     #plotting model accuracy graph 
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])                                                    #plotting model loss graph 
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


