import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
import soundfile as sf
import os
import numpy as np


def record_audio():
    st.set_page_config(page_title="Audio Recorder", page_icon=":microphone:")
    st.title("Audio Recorder")

    class AudioRecorder(VideoTransformerBase):
        def __init__(self):
            self.audio_frames = []

        def recv(self, frame):
            self.audio_frames.append(frame.to_ndarray())
            return frame

    webrtc_ctx = webrtc_streamer(key="audio-recorder", video_processor_factory=AudioRecorder,
                                 audio_receiver_size=256,
                                 client_settings=ClientSettings(
                                     rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                                     media_stream_constraints={"video": False, "audio": True},
                                 ))
    recorder = webrtc_ctx.video_processor
    if webrtc_ctx.state.playing:
        if len(recorder.audio_frames) > 0:
            filename = os.path.join("C:/Users/busse/Neuer Ordner (6)/Recognizer/tempDir2", "my_recording.wav")
            audio_data = np.hstack(recorder.audio_frames)
            sf.write(filename, audio_data, 48000)
            st.audio(filename)


if __name__ == "__main__":
    try:
        record_audio()
    except Exception as e:
            print(e)