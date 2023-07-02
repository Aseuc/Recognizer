
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn as sk
from io import StringIO
import io
import pickle
from sklearn import datasets
from sklearn import svm
import altair as alt
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import wave
import seaborn as sns
import sys
import pandas as pd
from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping 
from sklearn.model_selection import train_test_split as ts 
from scipy.io import wavfile
import librosa
from sklearn.svm import SVC as svc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from openpyxl import Workbook
import wave
import random
from pydub import AudioSegment
import pandas as pd
from openpyxl import load_workbook
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
from sklearn.metrics import accuracy_score
import os
from pydub import AudioSegment
import os
import tensorflow as tf
import tqdm
from keras.models import Sequential
from keras.layers  import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import soundfile as sf
import librosa.display
from scipy import signal
import tempfile
import openpyxl
from keras.regularizers import l1_l2
import pages.VoiceChoice as vc
import ballons_blue as bb
import ballons_red  as br
import randomFacts as rf



def mp4_to_wav(mp4_file, wav_file):
    audio = AudioSegment.from_file(mp4_file, format="mp4")
    audio.export(f"tempDir/{wav_file}", format="wav")

def get_current_date_time():
    now = datetime.now()
    date_time_str =now.strftime("%d_%m_%Y_%H_%M_%S_%f_%Z")
    return date_time_str + "_"

def upload_and_convert():
    uploaded_file = st.file_uploader("Wählen Sie eine Datei zum Hochladen aus", type=["mp4", "wav"], key="file_uploader3")
    print(uploaded_file)
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
        st.write(file_details)
        with open(os.path.join("tempDir2",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())
        if uploaded_file.type == "video/mp4":
            mp4_to_wav(uploaded_file,get_current_date_time() + ".wav")
            return True
        else:
            st.success("Hochgeladene Datei ist bereits im WAV-Format")
            return True


st.set_page_config(
    page_title="VoiceChoice - Random Forest Classifier!",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="auto"
)


@st.cache_data
def load_data(excel_file):
    df = pd.read_excel(excel_file)
    return df

try:
    if upload_and_convert() == True: 
        for file in os.listdir("tempDir2/"):
            if file.endswith(".wav"):
                excel_File = vc.get_single_excel_with_features_no_label(f"tempDir2/{file}",f"tempDir2/",1,False)
                # st.write(excel_File)
                
                df = load_data("TrainDataRFohneID1.xlsx")
                X = df.drop(["label"],axis=1)
                y = df["label"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                rfc = RandomForestClassifier(n_estimators=100, random_state=42)
                rfc.fit(X_train, y_train)
                y_pred = rfc.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                # with open('ML_Models/RandomForestClassifier.pkl', 'wb') as file:
                #      pickle.dump(rfc, file)
                
                
                st.write(f"Die Genauigkeit des Random Forest Klassifikators auf den Testdatensatz beträgt: {accuracy:.2f}")   
                st.write("Hier nochmal genauer die Predictions der Testdatensätze 0 = Mann, 1 = Frau:" , y_pred)

                # with open('ML_Models/RandomForestClassifier.pkl', 'rb') as file:
                #         rfc = pickle.load(file)
                data = pd.read_excel(excel_File)
                data = data.drop(["Unnamed: 0"], axis=1)            
                st.write(data)
                data = data.drop(["label"], axis=1)
                st.write(data)
                y_pred2 = rfc.predict(data)
                st.write(y_pred2)

                

                # df = load_data("TrainDataRFohneID1.xlsx")
                # X = df.drop(["label"], axis=1)
                # y = df["label"]
                # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                # rfc = RandomForestClassifier(random_state=42)
                # param_grid = {
                #     'n_estimators': [10, 50, 100, 200],
                #     'max_depth': [None, 10, 20, 30],
                #     'min_samples_split': [2, 5, 10],
                #     'min_samples_leaf': [1, 2, 4]
                # }
                # grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

                # grid_search.fit(X_train, y_train)
                
                # st.write(f"Die besten Hyperparameter sind: {grid_search.best_params_}")
                
                # best_model = grid_search.best_estimator_
                
                # y_pred = best_model.predict(X_test)
                
                # accuracy = accuracy_score(y_test, y_pred)
                
                # st.write(f"Die Genauigkeit des Random Forest Klassifikators auf den Testdatensatz beträgt: {accuracy:.2f}")
                # st.write("Hier nochmal genauer die Predictions der Testdatensätze 0 = Mann, 1 = Fraue:", y_pred)

                if y_pred2 == 0: 
                    bb.ballons_blue()
                    st.markdown("<h3 style='text-align: center;'>Die Person auf der Aufnahme scheint ein Mann zu sein!</h3>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: center;'>Wusstest du schon?</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='text-align: center;'>{rf.random_fact_men()}</h4>", unsafe_allow_html=True)
                elif y_pred2 == 1:
                    br.ballons_red()
                    st.markdown("<h3 style='text-align: center;'>Die Person auf der Aufnahme scheint eine Frau zu sein!</h3>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: center;'>Wusstest du schon?</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='text-align: center;'>{rf.random_fact_women()}</h4>", unsafe_allow_html=True)



                st.balloons()    
                for file in os.listdir("tempDir2/"):
                            if file.endswith(".wav"):    
                                        os.remove(os.path.join("tempDir2", file))
                            if file.endswith(".xlsx"):
                                    os.remove(os.path.join("tempDir2",(file)))
                            if file.endswith(".mp4"):
                                    os.remove(os.path.join("tempDir2",(file)))
except Exception as e:
        print(e)







