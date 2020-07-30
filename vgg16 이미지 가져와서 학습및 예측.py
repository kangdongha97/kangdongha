from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=None):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

base_model = vgg16.VGG16(weights='imagenet', include_top=True)
base_model.summary()

cat_dog = preprocess_image('cat_dog.jpg', target_size=(224, 224))
preds = base_model.predict(cat_dog)
print('Predicted:', vgg16.decode_predictions(preds, top=3)[0])
image = plt.imread('cat_dog.jpg')
plt.imshow(image)
plt.show()