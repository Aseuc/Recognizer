import streamlit as st
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings, WebRtcMode
from pydub import AudioSegment
import os
import numpy as np
import VoiceChoice as vc
import pandas as pd
from datetime import datetime, timedelta


def shuffle_excel_rows(filename, sheet_name=0):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(filename, sheet_name=sheet_name)

    # Shuffle the rows of the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    # Write the shuffled DataFrame back to the Excel file
    df.to_excel(filename, sheet_name=sheet_name, index=False)


def record_audio():
    try:
        st.set_page_config(page_title="VoiceChoice NN & Echtzeitaufnahme", page_icon="favicon.ico", layout="wide")
    except Exception as e:
        print(e)
    st.title('VoiceChoice mit NN & "Echzeitaufnahme"')

    def recorder_factory():
        return MediaRecorder("tempDir2/record.wav")

    class AudioRecorder(VideoTransformerBase):
        def __init__(self):
            self.audio_frames = []

        def recv(self, frame):
            self.audio_frames.append(frame.to_ndarray())
            return frame

    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        in_recorder_factory=recorder_factory,
        client_settings=ClientSettings(
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "audio": True,
                "video": False,
            },
        ),
    )

    recorder = webrtc_ctx.audio_processor

    start_time = datetime.now()
    timer_text = st.empty()

    if webrtc_ctx.state.playing:
        current_time = datetime.now()
        elapsed_time = current_time - start_time
        timer_text.text(f"Elapsed time: {str(elapsed_time)[:-7]}")

    if st.button("Neuronales Netz starten"):
        excel_file = vc.get_single_excel_with_features_no_label("tempDir2/record.wav", "tempDir2/", 10, True)
        vc.neuronal_network("TDNN3.xlsx", excel_file, 4, [32, 32, 32, 32])
        for file in os.listdir("tempDir2/"):
            if file.endswith(".xlsx"):
                os.remove(f"tempDir2/{file}")
            if file.endswith(".wav"):
                os.remove(f"tempDir2/{file}")


if __name__ == "__main__":
    try:
        record_audio()
    except Exception as e:
        print(e)
