import streamlit as st
import pandas as pd

# Kelas untuk menangani input pengguna, baik secara manual maupun dari file 
class InputHandler:
    def __init__(self):
        self.uploaded_file = None  # Menyimpan file csv yang diunggah pengguna
        self.input_df = None  # Menyimpan DataFrame input pengguna

    def user_input_features(self):
        # Mengambil input pengguna dari sidebar untuk tiap fitur
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen')) # Dropdown untuk memilih pulau tempat penguin ditemukan
        sex = st.sidebar.selectbox('Sex', ('male', 'female')) # Dropdown untuk memilih jenis kelamin penguin
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9) # Slider untuk mengatur panjang paruh
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2) # Slider untuk mengatur kedalaman paruh
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0) # Slider untuk mengatur panjang sirip
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0) # Slider untuk mengatur massa tubuh

        # Menggabungkan input ke dalam DataFrame
        data = {
            'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex
        }

        # Mengubah dictionary menjadi DataFrame dengan satu baris
        return pd.DataFrame(data, index=[0])

    def load_input(self):
        # Fungsi untuk memuat input pengguna dari file atau manual
        # Menampilkan uploader file di sidebar
        self.uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"]) 
        if self.uploaded_file is not None:
            # Jika ada file yang diunggah, baca sebagai DataFrame
            self.input_df = pd.read_csv(self.uploaded_file)
        else:
            # Jika tidak ada file, gunakan input manual dari user_input_features()
            self.input_df = self.user_input_features()
        return self.input_df # Mengembalikan DataFrame input pengguna