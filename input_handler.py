import streamlit as st
import pandas as pd

class InputHandler:
    def __init__(self):
        self.uploaded_file = None

    def user_input_features(self):
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)

        data = {
            'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex
        }
        return pd.DataFrame(data, index=[0])

    def load_input(self):
        self.uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        if self.uploaded_file:
            return pd.read_csv(self.uploaded_file)
        return self.user_input_features()