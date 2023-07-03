import streamlit as st
from moviepy.editor import AudioFileClip
import os
from pydub import AudioSegment

try:
    st.set_page_config(
        page_title="Converter",
        page_icon="favicon.ico",
        layout='wide',
        initial_sidebar_state="collapsed"
    )

    st.markdown("<div>Hier hast du die Möglichkeit MP3-Dateien hochzuladen und als .wav wieder herunterzuladen, "
                "falls du keine .wav-Datei hast.</div>", unsafe_allow_html=True)


    def convert_audio(file):
        filename, file_extension = os.path.splitext(file.name)
        if file_extension == '.mp3':
            with open(file.name, 'wb') as f:
                f.write(file.getvalue())
            audio = AudioSegment.from_file(file.name, format=file_extension[1:])
            wav_filename = filename + '.wav'
            audio.export(wav_filename, format='wav')
            st.success(f'Konvertiert nach {wav_filename}')
            with open(wav_filename, 'rb') as f:
                st.download_button(label='Herunterladen', data=f.read(), file_name=wav_filename, mime='audio/wav')
        else:
            st.error('Nur mp3 Dateien werden unterstützt.')


    uploaded_file = st.file_uploader('Wählen Sie eine Datei aus', type=['mp3'], key="8")
    if uploaded_file is not None:
        convert_audio(uploaded_file)
except Exception as e:
    print(e)

try:

    st.header("   ")
    st.markdown("<div>Hier hast du die Möglichkeit WhatsApp-Sprachnotizen hochzuladen und als .wav wieder "
                "herunterzuladen, falls du keine .wav-Datei hast.</div>", unsafe_allow_html=True)


    def convert_audio(file):
        filename, file_extension = os.path.splitext(file.name)
        if file_extension == '.opus':
            with open(file.name, 'wb') as f:
                f.write(file.getvalue())
            audio = AudioSegment.from_ogg(file.name)
            wav_filename = filename + '.wav'
            audio.export(wav_filename, format='wav')
            st.success(f'Konvertiert nach {wav_filename}')
            with open(wav_filename, 'rb') as f:
                st.download_button(label='Herunterladen', data=f.read(), file_name=wav_filename, mime='audio/wav')
        else:
            st.error('Nur opus Dateien werden unterstützt.')


    uploaded_file = st.file_uploader('Wählen Sie eine Datei aus', type=['opus'])
    if uploaded_file is not None:
        convert_audio(uploaded_file)
except Exception as e:
    print(e)

try:
    st.header("   ")
    st.markdown(
        "<div>Hier hast du die Möglichkeit MP4-Dateien hochzuladen und als .wav wieder herunterzuladen, "
        "falls du keine .wav-Datei hast.</div>",
        unsafe_allow_html=True)


    def convert_audio(file):
        filename, file_extension = os.path.splitext(file.name)
        if file_extension == '.mp4':
            with open(file.name, 'wb') as f:
                f.write(file.getvalue())
            audio = AudioFileClip(file.name)
            wav_filename = filename + '.wav'
            audio.write_audiofile(wav_filename)
            st.success(f'Konvertiert nach {wav_filename}')
            with open(wav_filename, 'rb') as f:
                st.download_button(label='Herunterladen', data=f.read(), file_name=wav_filename, mime='audio/wav')
        else:
            st.error('Nur mp4 Dateien werden unterstützt.')


    uploaded_file = st.file_uploader('Wählen Sie eine Datei aus', type=['mp4'], key="9")
    if uploaded_file is not None:
        convert_audio(uploaded_file)
except Exception as e:
    print(e)
