# Import library yang dibutuhkan
import streamlit as st
import pandas as pd

# Class untuk menangani input dari pengguna
class InputHandler:
    def __init__(self):
        self.uploaded_file = None   # Menyimpan file yang di-upload oleh pengguna

    # Fungsi untuk mengambil input manual dari sidebar Streamlit
    def user_input_features(self):
        # Pengguna memilih nilai untuk masing-masing fitur
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))   # Pilih pulau
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))                       # Pilih jenis kelamin
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9)    # Panjang paruh
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)      # Kedalaman paruh
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)   # Panjang sirip
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)    # Berat tubuh

        # Menyimpan semua input ke dalam dictionary
        data = {
            'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex
        }

        # Mengubah dictionary ke dalam format DataFrame
        return pd.DataFrame(data, index=[0])

    # Fungsi untuk membaca input dari file CSV atau form manual
    def load_input(self):
        # Pengguna bisa mengunggah file CSV sebagai input
        self.uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

        # Jika file diunggah, baca isinya
        if self.uploaded_file:
            return pd.read_csv(self.uploaded_file)
        
        # Jika tidak ada file yang diunggah, gunakan input manual
        return self.user_input_features()