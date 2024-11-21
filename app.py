import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

# Parameter model dan gambar
img_height, img_width = 64, 64

# Fungsi preprocess gambar
def preprocess_image(img):
    img = img.convert('L')  # Ubah ke grayscale
    img = img.resize((img_height, img_width))  # Resize gambar
    img_array = np.array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=-1)  # Tambahkan channel grayscale
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch size
    return img_array

# Load model terlatih
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('braille_model.h5')
    return model

model = load_model()

# Fungsi prediksi
def predict_braille_image(img, model):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_label = chr(97 + class_index)  # Konversi ke huruf a-z
    return class_label

# Streamlit UI
st.title("Braille Character Recognition")
st.write("Unggah gambar pola Braille dan aplikasi akan memprediksi hurufnya.")

# Input gambar dari pengguna
uploaded_file = st.file_uploader("Unggah gambar Braille (format: .jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Prediksi gambar
    if st.button("Prediksi"):
        predicted_label = predict_braille_image(img, model)
        st.success(f"Huruf yang terdeteksi: {predicted_label}")
