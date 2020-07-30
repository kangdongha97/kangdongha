# import libraries
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, callbacks, optimizers, models
from sklearn.metrics import classification_report
import sklearn

test_datagen = ImageDataGenerator(rescale=1./255)

ishape=48 # 224  48
test_digits_loader = test_datagen.flow_from_directory( 'data/test', target_size=(ishape, ishape),
        color_mode = 'rgb',  batch_size=1,  shuffle = False, class_mode='categorical')

from keras.models import load_model
model = load_model('model.h5')
output = model.predict_generator(test_digits_loader, steps=10)
print(output)
