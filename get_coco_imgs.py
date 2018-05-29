# get_features.py
import numpy as np
from keras.preprocessing import image
from keras.models import Model

from vgg16_hybrid_places_1365 import VGG16_Hubrid_1365
from places_utils import preprocess_input

loc_model = VGG16_Hubrid_1365(weights='places', include_top=True)


model = Model(loc_model.input, loc_model.get_layer('drop_fc2').output)

img_path = 'restaurant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
print features
print features.shape