# Mengimpor library Streamlit dan class dari modul lain
import streamlit as st
from models import InputHandler, Encoder, ModelLoader, Predictor

# Judul website
st.title("\U0001F427 Aplikasi Prediksi Spesies Penguin")
st.markdown("Masukkan data penguin dan lihat hasil prediksi spesiesnya.")

# Menggunakan blok try-except untuk menangani potensi error
try:
    # Inisialisasi handler untuk mengambil input dari pengguna
    input_handler = InputHandler()
    input_df = input_handler.load_input()

    # Encode input agar cocok dengan format model
    encoder = Encoder(input_df)
    encoded_df = encoder.encode_input()

    # Menampilkan data input setelah diproses (encode)
    st.subheader("Fitur Input Pengguna")
    st.write(encoded_df)

    # Memuat model yang sudah dilatih atau membangunnya jika belum ada
    model_loader = ModelLoader()
    model = model_loader.load_model()

    # Membuat prediksi berdasarkan input pengguna
    predictor = Predictor(model, encoded_df)
    prediction, prediction_proba = predictor.predict()

    # Menampilkan hasil prediksi dan probabilitasnya
    predictor.show_result(prediction, prediction_proba)

# Menangani jika terjadi kesalahan dan menampilkan pesan error di UI
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")