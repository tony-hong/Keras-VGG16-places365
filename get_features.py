# get_features.py
from vgg16_hybrid_places_1365 import VGG16_Hubrid_1365
from keras.preprocessing import image
from places_utils import preprocess_input
import numpy as np

model = VGG16_Hubrid_1365(weights='places', include_top=False)

img_path = 'restaurant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
print features