import streamlit as st
from input_handler import InputHandler
from encoder import Encoder
from model_loader import ModelLoader
from predictor import Predictor

# Header
st.write("""
# Prediksi Spesies Penguin Palmer

🐧 Aplikasi Prediksi Spesies Penguin 🐧

Aplikasi yang akan membantu kamu memprediksi spesies penguin berdasarkan ciri-ciri fisiknya.

Data: [Palmer Penguins - Kaggle](https://www.kaggle.com/datasets/ashkhagan/palmer-penguins-datasetalternative-iris-dataset/data)
""")

st.sidebar.header('Fitur Input Variabel')
st.sidebar.markdown("Silakan masukkan detail karakteristik penguin di bawah ini untuk memprediksi spesiesnya.")

# Ambil input pengguna
input_handler = InputHandler()
input_df = input_handler.load_input()

# Encode input
encoder = Encoder(input_df, 'C:/Users/User/PASD/Dataset/penguins_cleaned.csv')
df = encoder.encode_input()

# Tampilkan input
st.subheader('Fitur Input Pengguna')
st.write(df)

# Load model
model_loader = ModelLoader('penguins_clf.pkl')
model = model_loader.load_model()

# Prediksi
predictor = Predictor(model, df)
prediction, prediction_proba = predictor.predict()

# Tampilkan hasil
predictor.show_result(prediction, prediction_proba)