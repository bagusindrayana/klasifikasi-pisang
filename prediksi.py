import requests
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Parameters
input_size = (224, 224) # Bisa kalian ganti
#define input shape
channel = (3,)
input_shape = input_size + channel
#define labels
labels = ['belum_matang', 'matang']

def preprocess(img,input_size):
    nimg = img.convert('RGB').resize(input_size, resample= 0)
    img_arr = (np.array(nimg))/255
    return img_arr
def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)


MODEL_PATH = 'model/medium_project/model.h5'
model = load_model(MODEL_PATH,compile=False)

# read image
im = Image.open('matang_4.jpg')
X = preprocess(im,input_size)
X = reshape([X])
y = model.predict(X)
print( labels[np.argmax(y)], np.max(y) )