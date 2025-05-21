import streamlit as st
from input_handler import InputHandler
from encoder import Encoder
from model_loader import ModelLoader
from predictor import Predictor

# Menampilkan header aplikasi di halaman utama Streamlit
st.write("""
# Prediksi Spesies Penguin Palmer

🐧 Aplikasi Prediksi Spesies Penguin 🐧

Aplikasi yang akan membantu kamu memprediksi spesies penguin berdasarkan ciri-ciri fisiknya.

Data: [Palmer Penguins - Kaggle](https://www.kaggle.com/datasets/ashkhagan/palmer-penguins-datasetalternative-iris-dataset/data)
""")

# Membuat header di sidebar untuk input dari pengguna
st.sidebar.header('Fitur Input Variabel')
st.sidebar.markdown("Silakan masukkan detail karakteristik penguin di bawah ini untuk memprediksi spesiesnya.")

input_handler = InputHandler() # Membuat instance dari kelas InputHandler untuk menangani input pengguna
input_df = input_handler.load_input() # Mengambil input pengguna

# Melakukan encoding pada input pengguna agar sesuai dengan format model
encoder = Encoder(input_df, 'C:/Users/User/PASD/Dataset/penguins_cleaned.csv')
df = encoder.encode_input()

# Menampilkan input yang telah diolah kepada pengguna di halaman utama
st.subheader('Fitur Input Pengguna')
st.write(df)

# Membuat instance dari kelas ModelLoader untuk memuat model machine learning
model_loader = ModelLoader('penguins_clf.pkl')
model = model_loader.load_model() # Memuat model klasifikasi dari file pickle

predictor = Predictor(model, df) # Membuat instance dari kelas Predictor untuk melakukan prediksi
prediction, prediction_proba = predictor.predict() # Melakukan prediksi spesies penguin serta mendapatkan probabilitasnya

# Menampilkan hasil prediksi dan probabilitas kepada pengguna
predictor.show_result(prediction, prediction_proba)