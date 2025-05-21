import streamlit as st
from input_handler import InputHandler
from encoder import Encoder
from model_loader import ModelLoader
from predictor import Predictor

st.title("\U0001F427 Aplikasi Prediksi Spesies Penguin")
st.markdown("Masukkan data penguin dan lihat hasil prediksi spesiesnya.")

try:
    input_handler = InputHandler()
    input_df = input_handler.load_input()

    encoder = Encoder(input_df)
    encoded_df = encoder.encode_input()

    st.subheader("Fitur Input Pengguna")
    st.write(encoded_df)

    model_loader = ModelLoader()
    model = model_loader.load_model()

    predictor = Predictor(model, encoded_df)
    prediction, prediction_proba = predictor.predict()
    predictor.show_result(prediction, prediction_proba)

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")