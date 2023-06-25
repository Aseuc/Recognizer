
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
import numpy as np
import wave
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


def extract_zcr(file_name):
    y, sr = librosa.load(file_name)
    zcr = librosa.feature.zero_crossing_rate(y)
    df_ZCR = pd.DataFrame(zcr)
    for i in range(df_ZCR.shape[1]):
        df_ZCR = df_ZCR.rename(columns={i: f"Zero Crossing Rate{i+1}"})
    return df_ZCR




def plot_zcr(file_name):
    y, sr = librosa.load(file_name)
    zcr = librosa.feature.zero_crossing_rate(y)


def plot_loudness(file_name):
    y, sr = librosa.load(file_name)
    S = np.abs(librosa.stft(y))
    loudness = librosa.feature.spectral_bandwidth(S=S)
    
    plt.figure()
    plt.plot(loudness[0])
    plt.title('Loudness')
    plt.show()
def extract_snr(file_name):
    y, sr = librosa.load(file_name)
    S = np.abs(librosa.stft(y))
    snr = librosa.feature.spectral_contrast(S=S)
    dfsnr = pd.DataFrame(snr)
    for i in range(dfsnr.shape[1]):
        dfsnr = dfsnr.rename(columns={i: f"Spektral Kontrast{i+1}"})
    return dfsnr
def plot_snr(file_name):
    snr = extract_snr(file_name)
    plt.figure(figsize=(10, 4))
    plt.imshow(snr, aspect='auto', origin='lower', cmap='coolwarm')
    plt.colorbar()
    plt.ylabel('Frequency bands')
    plt.xlabel('Time (frames)')
    plt.title('Spectral contrast')
    plt.tight_layout()
    plt.show()

def extract_bandwidth(file_name):
    y, sr = librosa.load(file_name)
    S = np.abs(librosa.stft(y))
    bandwidth = librosa.feature.spectral_bandwidth(S=S)
    dfBandwith = pd.DataFrame(bandwidth)
    for i in range(dfBandwith.shape[1]):
        dfBandwith = dfBandwith.rename(columns={i: f"Bandbreite{i+1}"})
    return dfBandwith
def plot_bandwidth(file_name):
    bandwidth = extract_bandwidth(file_name)
    plt.figure(figsize=(10, 4))
    plt.imshow(bandwidth, aspect='auto', origin='lower', cmap='coolwarm')
    plt.colorbar()
    plt.ylabel('Frequency bands')
    plt.xlabel('Time (frames)')
    plt.title('Spectral bandwidth')
    plt.tight_layout()
    plt.show()
    

def get_current_date_time():
    now = datetime.now()
    date_time_str =now.strftime("%d_%m_%Y_%H_%M_%S_%f_%Z")
    return date_time_str + "_"

def split_wav(destinationPath, wav_file, segment_length=3000):
    audio = AudioSegment.from_wav(wav_file)

    for i in range(0, len(audio), segment_length):
        segment = audio[i:i+segment_length]
        segment.export(f"{destinationPath}segment__{i-10//segment_length}.wav", format="wav")
        

def getDuration(input_file_path,duration):
    with wave.open(input_file_path, 'rb') as input_wav:
        n_channels = input_wav.getnchannels()
        sample_width = input_wav.getsampwidth()
        frame_rate = input_wav.getframerate()
        n_frames = input_wav.getnframes()

        n_frames_duration = int(frame_rate * duration)
        return n_frames_duration 
    
def get_n_frames_duration (input_file_path,duration):
    with wave.open(input_file_path, 'rb') as input_wav:
        n_channels = input_wav.getnchannels()
        sample_width = input_wav.getsampwidth()
        frame_rate = input_wav.getframerate()
        n_frames = input_wav.getnframes()

        n_frames_duration = int(frame_rate * duration)
        return n_frames_duration 
    
# Falls diese Funktion und die split_multiple_frames() nicht mehr funktioniert lösche nameOfXLSX
def extract_random_sequence(input_file_path, output_file_path,duration=3):
    with wave.open(input_file_path, 'rb') as input_wav:
        n_channels = input_wav.getnchannels()
        sample_width = input_wav.getsampwidth()
        frame_rate = input_wav.getframerate()
        n_frames = input_wav.getnframes()

        n_frames_duration = int(frame_rate * duration)

        if n_frames_duration > n_frames:
            raise ValueError(f"Die angegebene Dauer ({duration} Sekunden) ist länger als die Gesamtdauer der Datei ({n_frames / frame_rate} Sekunden).")

        start_frame = random.randint(0, n_frames - n_frames_duration)

        input_wav.setpos(start_frame)
        frames = input_wav.readframes(n_frames_duration)
        

    with wave.open(f"{output_file_path}"+".wav", 'wb') as output_wav:
        output_wav.setparams((n_channels, sample_width, frame_rate, n_frames_duration, 'NONE', 'not compressed'))
        output_wav.writeframes(frames)


def get_n_frames(input_file_path):
      with wave.open(input_file_path, 'rb') as input_wav:
        n_channels = input_wav.getnchannels()
        sample_width = input_wav.getsampwidth()
        frame_rate = input_wav.getframerate()
        n_frames = input_wav.getnframes()

        # n_frames_duration = int(frame_rate * duration)
        return n_frames 


def split_multiple_frames(input_file_path,output_file_path,duration=3):
    i = 0
    for files in os.listdir(input_file_path):

        if files.endswith(".wav"):
                
            # if get_n_frames_duration("longMenAudios/" + files,duration=3) > get_n_frames("longMenAudios/" + files):
                extract_random_sequence(input_file_path+ files,output_file_path  + f"{get_current_date_time()}duration_{duration}",duration=3)
                
                print(f"Random{i}duration_{duration}")
                i = i+1



def rename_data(file_path="MenSequences" or "WomenSequences",files_to_rename="Random"):
    i = 0
    for files in os.listdir(file_path):
        if files.endswith(".wav"):
            new_name = f"{files_to_rename}_{i}.wav"
            os.rename(os.path.join(file_path, files), os.path.join(file_path, new_name))
            i += 1


def upload_and_convert():
    uploaded_file = st.file_uploader("Wählen Sie eine Datei zum Hochladen aus", type=["mp4", "wav"], key="file_uploader")
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
        st.write(file_details)
        with open(os.path.join("tempDir",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())
        if uploaded_file.type == "audio/mp4":
            mp4_audio = AudioSegment.from_file(os.path.join("tempDir",uploaded_file.name), format="mp4")
            wav_filename = os.path.splitext(uploaded_file.name)[0] + ".wav"
            mp4_audio.export(os.path.join("tempDir",wav_filename), format="wav")
            st.success(f"Konvertierte Datei: {wav_filename}")
        else:
            st.success("Hochgeladene Datei ist bereits im WAV-Format")
            return True
    else:
        for file in os.listdir("tempDir"):
            if file.endswith(".wav"):
                os.remove(os.path.join("tempDir", file))
                return False





def get_features_df_csv(nameOfCsv,ordner_path):
    i = 0
    for file in os.listdir(ordner_path):
        print(file)
        if file.endswith('.wav'):
            newPath = ordner_path + "/" + file
            df_mfcc2 = extract_mfcc(newPath)
            df_loudness2 = extract_loudness(newPath)
            df_snr2 = extract_snr(newPath)
            df_zcr2 = extract_zcr(newPath)
            df_bandwith2 = extract_bandwidth(newPath)
            df_bandwith2 = pd.DataFrame(df_bandwith2)
            df_loudness2 = pd.DataFrame(df_loudness2)
            df_snr2 = pd.DataFrame(df_snr2)
            df_zcr2 = pd.DataFrame(df_zcr2)
            df_mfcc2 = pd.DataFrame(df_mfcc2)
            df_bandwith2 = df_bandwith2.iloc[:, :80]
            df_loudness2 = df_loudness2.iloc[:, :80]
            df_snr2 = df_snr2.iloc[:, :80]
            df_zcr2 = df_zcr2.iloc[:, :80]
            df_mfcc2 = df_mfcc2.iloc[:, :80]

            bandwidth_vec2 = df_bandwith2.values.flatten()
            zcr_vec2 = df_zcr2.values.flatten()
            sc_vec2 = df_snr2.values.flatten()
            mfcc_vec2 = df_mfcc2.values.flatten()
            chroma_vec2 = df_loudness2.values.flatten()

            df_bandwith2 = pd.DataFrame(index=[0], columns=['Bandbreite'])
            df_loudness2 = pd.DataFrame(index=[1], columns= ['Tonstärke'])
            df_snr2 = pd.DataFrame(index=[2], columns=   ['Spektral Kontrast']) 
            df_zcr2 = pd.DataFrame(index=[3], columns= ['Zero Crossing Rate'])
            df_mfcc2 = pd.DataFrame(index=[4], columns= ['MFCC'])


            df_bandwith2.at[0, 'Bandbreite'] = bandwidth_vec2.tolist()
            df_loudness2.at[1,'Tonstärke'] = chroma_vec2.tolist()
            df_snr2.at[2, 'Spektral Kontrast'] = sc_vec2.tolist()
            df_zcr2.at[3, 'Zero Crossing Rate'] = zcr_vec2.tolist()
            df_mfcc2.at[4, 'MFCC'] = mfcc_vec2.tolist()

            df_bandwith2['id']=range(1,len(df_bandwith2)+1)
            df_loudness2['id']=range(1,len(df_loudness2)+1)
            df_snr2['id']=range(1,len(df_snr2)+1)
            df_zcr2['id']=range(1,len(df_zcr2)+1)
            df_mfcc2['id']=range(1,len(df_mfcc2)+1)



            mergeFirst = pd.merge(df_mfcc2,df_zcr2,on='id')
            mergeSecond = pd.merge(mergeFirst, df_loudness2, on='id')
            mergeThird = pd.merge(mergeSecond, df_snr2,on='id')
            mergeForth = pd.merge(mergeThird, df_bandwith2,on='id')
            mergeForth['Frau/Mann'] = ['Frau']
            mergeForth.to_csv(f"{nameOfCsv}{i}.csv")

            
            i+=1
def extract_mfcc(file_name, n_mfcc=13):
    y, sr = librosa.load(file_name)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    df_MFCC = pd.DataFrame(mfcc)
    for i in range(df_MFCC.shape[1]):
        df_MFCC = df_MFCC.rename(columns={i: f"MFCC{i+1}"})
    return df_MFCC

def extract_loudness(file_name):
    y, sr = librosa.load(file_name)
    S = np.abs(librosa.stft(y))
    loudness = librosa.feature.spectral_bandwidth(S=S)
    df_Loudness = pd.DataFrame(loudness)
    for i in range(df_Loudness.shape[1]):
        df_Loudness = df_Loudness.rename(columns={i: f"Tonstärke{i+1}"})
    return df_Loudness

def plot_loudness(file_name):
    y, sr = librosa.load(file_name)
    S = np.abs(librosa.stft(y))
    loudness = librosa.feature.spectral_bandwidth(S=S)
def get_features_df_excel(ordner_path, destinationPath, nameOfXLSX, numberOfXLSXData, labelType, numberOfColumns):
    i = 0
    for file in os.listdir(ordner_path):
        if i == numberOfXLSXData: 
            return
        if file.endswith('.wav'):
            newPath = ordner_path + "/" + file
            df_mfcc2 = extract_mfcc(newPath)
            df_loudness2 = extract_loudness(newPath)
            df_snr2 = extract_snr(newPath)
            df_zcr2 = extract_zcr(newPath)
            df_bandwith2 = extract_bandwidth(newPath)
            df_bandwith2 = pd.DataFrame(df_bandwith2)
            df_loudness2 = pd.DataFrame(df_loudness2)
            df_snr2 = pd.DataFrame(df_snr2)
            df_zcr2 = pd.DataFrame(df_zcr2)
            df_mfcc2 = pd.DataFrame(df_mfcc2)
            df_bandwith2 = df_bandwith2.iloc[:, :numberOfColumns]
            df_loudness2 = df_loudness2.iloc[:, :numberOfColumns]
            df_snr2 = df_snr2.iloc[:, :numberOfColumns]
            df_zcr2 = df_zcr2.iloc[:, :numberOfColumns]
            df_mfcc2 = df_mfcc2.iloc[:, :numberOfColumns]

    
            df_bandwith2['id']=range(1,len(df_bandwith2)+1)
            df_loudness2['id']=range(1,len(df_loudness2)+1)
            df_snr2['id']=range(1,len(df_snr2)+1)
            df_zcr2['id']=range(1,len(df_zcr2)+1)
            df_mfcc2['id']=range(1,len(df_mfcc2)+1)



            mergeFirst = pd.merge(df_mfcc2,df_zcr2,on='id')
            mergeSecond = pd.merge(mergeFirst, df_loudness2, on='id')
            mergeThird = pd.merge(mergeSecond, df_snr2,on='id')
            mergeForth = pd.merge(mergeThird, df_bandwith2,on='id')
            mergeForth.dropna()
            
            num_rows = mergeForth.shape[0]
  
            for i in range(num_rows):
                mergeForth.at[i,'id'] = i
                i = i+1
                mergeForth = mergeForth.rename(columns={'Unnamed: 0': 'ID'})
                mergeForth.dropna()
                mergeForth = mergeForth.drop("id", axis=1)
                # mergeForth.head()
                # mergeForth = mergeForth.sample(frac=1).reset_index(drop=True)
                if labelType == "Frau" or labelType == "Mann" or labelType == 1 or labelType == 0:
                    mergeForth['label'] = [f'{labelType}']
                    mergeForth.to_excel(f"{destinationPath}{nameOfXLSX}{i}.xlsx")
               
            
        i = i+1


                        
def get_features_from_single_file_df_excel(input_ordner_path, destinationPath, nameOfWAVFile, nameOfXLSX, labelType, numberOfColumns):
    i = 0
    for file in os.listdir(input_ordner_path):
        if file == nameOfWAVFile:
            newPath = input_ordner_path + "/" + file
            df_mfcc2 = extract_mfcc(newPath)
            df_loudness2 = extract_loudness(newPath)
            df_snr2 = extract_snr(newPath)
            df_zcr2 = extract_zcr(newPath)
            df_bandwith2 = extract_bandwidth(newPath)
            df_bandwith2 = pd.DataFrame(df_bandwith2)
            df_loudness2 = pd.DataFrame(df_loudness2)
            df_snr2 = pd.DataFrame(df_snr2)
            df_zcr2 = pd.DataFrame(df_zcr2)
            df_mfcc2 = pd.DataFrame(df_mfcc2)
            df_bandwith2 = df_bandwith2.iloc[:, :numberOfColumns]
            df_loudness2 = df_loudness2.iloc[:, :numberOfColumns]
            df_snr2 = df_snr2.iloc[:, :numberOfColumns]
            df_zcr2 = df_zcr2.iloc[:, :numberOfColumns]
            df_mfcc2 = df_mfcc2.iloc[:, :numberOfColumns]    
            df_bandwith2['id']=range(1,len(df_bandwith2)+1)
            df_loudness2['id']=range(1,len(df_loudness2)+1)
            df_snr2['id']=range(1,len(df_snr2)+1)
            df_zcr2['id']=range(1,len(df_zcr2)+1)
            df_mfcc2['id']=range(1,len(df_mfcc2)+1)
            mergeFirst = pd.merge(df_mfcc2,df_zcr2,on='id')
            mergeSecond = pd.merge(mergeFirst, df_loudness2, on='id')
            mergeThird = pd.merge(mergeSecond, df_snr2,on='id')
            mergeForth = pd.merge(mergeThird, df_bandwith2,on='id')
            mergeForth.dropna()
            num_rows = mergeForth.shape[0]
            for i in range(num_rows):
                mergeForth.at[i,'id'] = i
                i = i+1
                mergeForth = mergeForth.rename(columns={'Unnamed: 0': 'ID'})
                mergeForth.dropna()
                mergeForth = mergeForth.drop("id", axis=1)
                # mergeForth.head()
                # mergeForth = mergeForth.sample(frac=1).reset_index(drop=True)
            if labelType == "Frau" or labelType == "Mann" or labelType == 0 or labelType == 1:
                    mergeForth['label'] = [f'{labelType}']
                    mergeForth.to_excel( f"{destinationPath}{nameOfXLSX}{i}" + ".xlsx")
                    print(destinationPath)
            else:
                 raise Exception("Label type not recognized!")
        i = i+1


def mp4_to_wav(mp4_file, wav_file):
    audio = AudioSegment.from_file(mp4_file, format="mp4")
    audio.export(wav_file, format="wav")



def split_wav(wav_file, segment_length=3000):
    audio = AudioSegment.from_wav(wav_file)
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i+segment_length]
        segment.export(f"segment_{i//segment_length}.wav", format="wav")





check = upload_and_convert()
if check == True:

    st.title("Extraktion der MFCC-Werte aus der WAV-Audio-Datei")
    st.write("MFCC (Mel Frequency Cepstral Coefficients): MFCCs werden zur automatischen Spracherkennung verwendet und führen zu einer kompakten Darstellung des Frequenzspektrums. ")

    for file in os.listdir("Recognizer/tree/main/tempDir"):
        if file.endswith(".wav"):
            st.write(file)
            df = extract_mfcc(f"Recognizer/tree/main/tempDir/{file}",n_mfcc=1)
            df = df.iloc[:, :10]
            st.write(df)
            
















