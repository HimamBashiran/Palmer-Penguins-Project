import streamlit as st
from input_handler import InputHandler
from encoder import Encoder
from model_loader import ModelLoader
from predictor import Predictor

# Header aplikasi
st.write("""
# Prediksi Spesies Penguin Palmer

🐧 Aplikasi Prediksi Spesies Penguin 🐧

Aplikasi yang akan membantu kamu memprediksi spesies penguin berdasarkan ciri-ciri fisiknya.

Data: [Palmer Penguins - Kaggle](https://www.kaggle.com/datasets/ashkhagan/palmer-penguins-datasetalternative-iris-dataset/data)
""")

# Sidebar untuk input dari pengguna
st.sidebar.header('Fitur Input Variabel')
st.sidebar.markdown("Silakan masukkan detail karakteristik penguin di bawah ini untuk memprediksi spesiesnya.")

# Mengambil input pengguna
input_handler = InputHandler()
input_df = input_handler.load_input()

# Melakukan encoding pada input pengguna
encoder = Encoder(input_df, 'C:/Users/User/PASD/Dataset/penguins_cleaned.csv')
df = encoder.encode_input()

# Menampilkan input yang dimasukkan pengguna
st.subheader('Fitur Input Pengguna')
st.write(df)

# Memuat model
model_loader = ModelLoader('penguins_clf.pkl')
model = model_loader.load_model()

# Melakukan prediksi
predictor = Predictor(model, df)
prediction, prediction_proba = predictor.predict()

# Menampilkan hasil prediksi
predictor.show_result(prediction, prediction_proba)