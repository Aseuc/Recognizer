import streamlit as st
from streamlit_webrtc import webrtc_streamer, ClientSettings, AudioProcessorBase, WebRtcMode
import soundfile as sf
import numpy as np

# st.set_page_config(page_title="Audioaufnahme", page_icon=":microphone:")
#
# st.title("Audioaufnahme")
#
#
# class AudioRecorder(AudioProcessorBase):
#     def __init__(self):
#         self.sample_rate = 16000
#         self.frames = []
#
#     def recv(self, frame):
#         self.frames.append(frame.to_ndarray())
#         return frame
#
#     def save_wav(self, filename):
#         data = np.concatenate(self.frames, axis=0)
#         sf.write(filename, data, self.sample_rate)
#
#
# recorder = AudioRecorder()
#
# webrtc_ctx = webrtc_streamer(key="sendonly-audio", mode=WebRtcMode.SENDONLY, audio_processor_factory=recorder, client_settings=ClientSettings(rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#                                  media_stream_constraints={
#                                      "audio": True,
#                                      "video": False,
#                                  },
#                              ),
#                              )
#
# if st.button("Aufnahme speichern"):
#     recorder.save_wav("tempDir2/record.wav")
#     st.write("Aufnahme gespeichert.")
