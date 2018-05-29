# predict.py

from vgg16_places_365 import VGG16_Places365
from keras.preprocessing import image
from places_utils import preprocess_input
import numpy as np
import os




model = VGG16_Places365(weights='places')
# kernel = model.get_layer('predictions').get_weights()[0]
# bias = np.zeros(365)
# model.get_layer('predictions').set_weights([kernel, bias])


img_path = '197.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

predictions_to_return = 10
preds = model.predict(x)[0]

print preds.sum()
print preds


top_preds = np.argsort(preds)[::-1][0:predictions_to_return]

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

print('--SCENE CATEGORIES:')
# output the prediction
for i in range(0, predictions_to_return):
    print(top_preds[i], classes[top_preds[i]])
