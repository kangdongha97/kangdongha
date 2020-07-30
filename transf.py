# import libraries
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, callbacks, optimizers, models
from sklearn.metrics import classification_report
import sklearn

train_datagen = ImageDataGenerator(rescale=1./255,)
test_datagen = ImageDataGenerator(rescale=1./255)

ishape=48 # 224  48
train_digits_loader = train_datagen.flow_from_directory('data/train', target_size=(ishape, ishape),
        batch_size=5, color_mode = 'rgb',  class_mode='categorical')

test_digits_loader = test_datagen.flow_from_directory( 'data/test', target_size=(ishape, ishape),
        color_mode = 'rgb',  batch_size=1,  shuffle = False, class_mode='categorical')


num_rows = 4
num_cols = 10
num_images = num_rows*num_cols
plt.figure(figsize=(2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2 *num_cols, 2*i + 1)
  img_path = 'data/train/'+train_digits_loader.filenames[i]
  a = Image.open(img_path)
  arr = np.array(a)
  # print(arr)
  plt.imshow(arr,cmap= 'gray')
plt.show()


from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.datasets import mnist
import cv2
import h5py as h5py 
import numpy as np

model_vgg = VGG16(include_top = False, weights = 'imagenet', input_shape = (ishape,ishape,3))
for layer in model_vgg.layers:
    layer.trainable = False
model = Flatten(name = 'flatten')(model_vgg.output) 
model = Dense(4096, activation='relu', name='fc1')(model)
# model = Dense(4096, activation='relu', name='fc2')(model)
model = Dropout(0.5)(model)
model = Dense(5, activation = 'softmax')(model) 
model_pretrain = Model(model_vgg.input, model, name = 'vgg16_pretrain')
model_pretrain.summary()
sgd = SGD(lr = 0.05, decay = 1e-5) 
model_pretrain.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

sample_size = train_digits_loader.n
print(train_digits_loader)
batch_size = 32
hist = model_pretrain.fit_generator( train_digits_loader,  steps_per_epoch=sample_size//batch_size,
   epochs=50, validation_data=test_digits_loader,   validation_steps=5)
#epochs =50
scores = model_pretrain.evaluate_generator(test_digits_loader, steps=5)
print(scores)

print("-- Predict --")
output = model_pretrain.predict_generator(test_digits_loader, steps=105)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_digits_loader.class_indices)
print(output)

from keras.models import load_model
model_pretrain.save('model.h5')