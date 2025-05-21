import numpy as np
import streamlit as st
from config import CLASS_LABELS

class Predictor:
    def __init__(self, model, input_df):
        self.model = model
        self.input_df = input_df

    def predict(self):
        prediction = self.model.predict(self.input_df)
        prediction_proba = self.model.predict_proba(self.input_df)
        return prediction, prediction_proba

    def show_result(self, prediction, prediction_proba):
        label = CLASS_LABELS[prediction[0]]
        st.subheader('Prediksi')
        st.write(f"Spesies yang diprediksi: **{label}**")

        images = {
            'Adelie': "https://i.pinimg.com/564x/e2/b3/cd/e2b3cd99d77364f751d8f42f81f1e78e.jpg",
            'Gentoo': "https://birdlifedata.blob.core.windows.net/species-images/22697755.jpg",
            'Chinstrap': "https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcQI2Eyelti4suqlwyxZK0_xRYDa8aCQMl0C4Ue00NlqFDEBLaRmtBQ1mhgr-XHtLUHyP0J63IlhJNdaY8axdrLr4A"
        }
        st.image(images[label], caption=f'Penguin {label}', use_container_width=True)

        st.subheader('Probabilitas Prediksi')
        st.write(prediction_proba)