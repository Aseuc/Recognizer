import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import VoiceChoice as vc

audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    with open("tempDir2/record.wav", "wb") as f:
        f.write(audio_bytes)

if st.button("Neuronales Netz Klassifizierung starten!"):
    excel_file = vc.get_single_excel_with_features_no_label("tempDir2/record.wav","tempDir2/",10,True)
    vc.neuronal_network("TDNN3.xlsx",excel_file)
    os.remove(excel_file)
    os.remove("tempDir2/record.wav")










