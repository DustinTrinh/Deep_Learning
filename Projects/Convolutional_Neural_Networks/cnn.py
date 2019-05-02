# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# PART 1 - Building the CNN
#import libraries
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Conv2D

#Initialising the CNN
classifier = Sequential()

#Step 1 - Convolution

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))



#Step 2 - Pooling -> Main purpose is to reduce size of the feature map and prevent overfitting
#Most of the time we use 2 by 2 for pool size
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding 2nd convolutional layer will help increase the accuracy
#and apply the pooling again
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 3 - Flatten -> Prepare the vector for the CONNECT step
classifier.add(Flatten())

#Step 4 - Full Connection
#Units stand for the nodes in hidden layer
classifier.add(Dense(units = 128, activation = 'relu'))

#add output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#PART 2
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#Target size is the size expected in the CNN model
#And since the input_shape is 64x64 (line 22), we will put target_size as 64,64
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#SInce we have 8000 images, the steps_per_epoch will be 8000
#epochs is number of training times. How many times do we want it to run ? 25
#validation_steps is the number of test_set --> 2000
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

#Testing the prediction
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
        prediction = 'dog'
else:
        prediction = 'cat'