from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

model = ResNet50(weights='imagenet')

img_path = 'cat_dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
model.summary()
preds = model.predict(x)
# 결과를 튜플의 리스트(클래스, 설명, 확률)로 디코딩합니다
# (배치 내 각 샘플 당 하나의 리스트)
print('Predicted:', decode_predictions(preds, top=3)[0])
# 예측결과: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
image = plt.imread(img_path)
plt.imshow(image)
plt.show()
