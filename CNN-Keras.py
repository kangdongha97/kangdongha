import numpy as np
from keras.datasets import mnist
import keras
import gc
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train[0].shape)
print(y_train[0])
X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32') 
X_train /= 255 
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
X_test /= 255
print(y_train[0])

def tran_y(y): 
    y_ohe = np.zeros(10) 
    y_ohe[y] = 1 
    return y_ohe

y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))]) 
y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])
y_train_ohe = y_train_ohe.astype('float32')
y_test_ohe = y_test_ohe.astype('float32')

model = Sequential() 
model.add(Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu',
          input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Dropout(0.5)) 
model.add(Conv2D(256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation = 'relu')) 
model.add(Dense(64, activation = 'relu')) 
model.add(Dense(32, activation = 'relu')) 
model.add(Dense(10, activation = 'softmax'))

adamoptimizer = keras.optimizers.Adam(lr = 1e-4)
model.compile(loss = 'categorical_crossentropy', 
              optimizer = adamoptimizer, metrics = ['accuracy'])

model.fit(X_train, y_train_ohe, validation_data = (X_test, y_test_ohe), 
          epochs = 20, batch_size = 128)