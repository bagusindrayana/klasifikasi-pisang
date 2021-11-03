import streamlit as st
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model


st.set_page_config(
	page_title="Kalsifikasi Pisang Yang Matang Dan Belum",
)
### Excluding Imports ###
st.title("Kalsifikasi Pisang Yang Matang Dan Belum")

uploaded_file = st.file_uploader("Pilih Gambar Pisang...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
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
    X = preprocess(image,input_size)
    X = reshape([X])
    y = model.predict(X)
    st.write(labels[np.argmax(y)], np.max(y))