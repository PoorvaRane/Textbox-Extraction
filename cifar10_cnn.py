from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import pickle
import numpy as np
from keras.callbacks import ModelCheckpoint


batch_size = 32
num_classes = 1
epochs = 1000

f_train = pickle.load(open('train_data.pkl','rb'))
f_test = pickle.load(open('validation_data.pkl','rb'))

f_train=np.array(f_train)
f_test=np.array(f_test)

xl_train=f_train[:,0]
y_train=f_train[:,1]

xl_test=f_test[:,0]
y_test=f_test[:,1]

x_train=np.zeros((len(xl_train),14,14))
for i in range(len(x_train)):
  x_train[i,:,:]=xl_train[i]

x_train=x_train.reshape(10400,14,14,1)

x_test=np.zeros((len(xl_test),14,14))
for i in range(len(x_test)):
  x_test[i,:,:]=xl_test[i]

x_test=x_test.reshape(800,14,14,1)


model = Sequential()
model.add(Conv2D(4,(4,4), padding='same',input_shape=(14,14,1)))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

model.summary()

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop()

# Let's train the model using RMSprop
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])


checkpointer = ModelCheckpoint(filepath='nnet.h5', 
                monitor='val_acc', verbose=1, save_best_only=True)
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,
          validation_data=(x_test, y_test),callbacks=[checkpointer])
