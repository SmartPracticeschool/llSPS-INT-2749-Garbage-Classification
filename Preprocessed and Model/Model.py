#Importing The Model Building Libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

#Initializing The Model
model = Sequential()

#Loading The Prepocessed Data
from Preprocessed import *

#Adding CNN Layers
model.add(Convolution2D(32,(3,3),input_shape = (64,64,3),activation = 'relu'))

#Configure The Learning Process
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())

#Adding Dense Layers
model.add(Dense(output_dim = 128 ,init = 'uniform',activation = 'relu'))
model.add(Dense(output_dim = 6,activation = 'softmax',init ='uniform'))

#Optimise The Model
model.compile(loss = 'categorical_crossentropy',optimizer = "adam",metrics = ["accuracy"])

#Train And Test The Model
model.fit_generator(x_train, steps_per_epoch = 126,epochs = 1000,validation_data = x_test,validation_steps = 32)

#Saving the model
model.save("garbage.h5")
