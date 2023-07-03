import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import VoiceChoice as vc


st.set_page_config(
    layout="wide",
    page_icon="favicon.ico",
    page_title="VoiceChoice - NN & Echzeitaufnahme",
    initial_sidebar_state="collapsed"

)


vc.add_logo_sidebar()
audio_bytes = audio_recorder("Mikrofon anklicken um Aufnahme zu starten!")

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    with open("tempDir2/record.wav", "wb") as f:
        f.write(audio_bytes)

if st.button("Neuronales Netz Klassifizierung starten!"):
    excel_file = vc.get_single_excel_with_features_no_label("tempDir2/record.wav","tempDir2/",10,True)
    vc.neuronal_network("TDNN3.xlsx",excel_file)
    os.remove(excel_file)
    os.remove("tempDir2/record.wav")










