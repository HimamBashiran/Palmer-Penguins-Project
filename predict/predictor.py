# Import Library yang dibutuhkan
import numpy as np
import streamlit as st
from config import CLASS_LABELS # Import Label kelas dari file config

# Membuat class Predictor untuk melakukan prediksi spesies penguin
class Predictor:
    # Inisialisasi class dengan model dan data input dari pengguna
    def __init__(self, model, input_df):
        self.model = model        # Model machine learning
        self.input_df = input_df  # Data yang dimasukkan oleh pengguna

    # Fungsi untuk melakukan prediksi
    def predict(self):
        prediction = self.model.predict(self.input_df)              # Prediksi spesies berdasarkan data input
        prediction_proba = self.model.predict_proba(self.input_df)  # Mendapatkan nilai probabilitas dari setiap spesies
        return prediction, prediction_proba                         # Mengembalikan hasil prediksi dan probabilitasnya
    
    # Fungsi untuk menampilkan hasil prediksi ke layar Streamlit
    def show_result(self, prediction, prediction_proba):
        label = CLASS_LABELS[prediction[0]] # Mengubah hasil prediksi (angka) menjadi nama spesies
        # Menampilkan hasil prediksi ke pengguna
        st.subheader('Prediksi')
        st.write(f"Spesies yang diprediksi: **{label}**")

        # Gambar dari masing-masing spesies penguin
        images = {
            'Adelie': "https://i.pinimg.com/564x/e2/b3/cd/e2b3cd99d77364f751d8f42f81f1e78e.jpg",
            'Gentoo': "https://birdlifedata.blob.core.windows.net/species-images/22697755.jpg",
            'Chinstrap': "https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcQI2Eyelti4suqlwyxZK0_xRYDa8aCQMl0C4Ue00NlqFDEBLaRmtBQ1mhgr-XHtLUHyP0J63IlhJNdaY8axdrLr4A"
        }

        # Menampilkan gambar penguin sesuai hasil prediksi
        st.image(images[label], caption=f'Penguin {label}', use_container_width=True)

        # Menampilkan nilai probabilitas dari masing-masing spesies
        st.subheader('Probabilitas Prediksi')
        st.write(prediction_proba)