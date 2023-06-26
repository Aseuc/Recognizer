
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


def extract_zcr(file_name):
    y, sr = librosa.load(file_name)
    zcr = librosa.feature.zero_crossing_rate(y)
    df_ZCR = pd.DataFrame(zcr)
    for i in range(df_ZCR.shape[1]):
        df_ZCR = df_ZCR.rename(columns={i: f"Zero Crossing Rate{i+1}"})
    return df_ZCR








def extract_snr(file_name):
    y, sr = librosa.load(file_name)
    S = np.abs(librosa.stft(y))
    snr = librosa.feature.spectral_contrast(S=S)
    dfsnr = pd.DataFrame(snr)
    for i in range(dfsnr.shape[1]):
        dfsnr = dfsnr.rename(columns={i: f"Spektral Kontrast{i+1}"})
    return dfsnr

def extract_bandwidth(file_name):
    y, sr = librosa.load(file_name)
    S = np.abs(librosa.stft(y))
    bandwidth = librosa.feature.spectral_bandwidth(S=S)
    dfBandwith = pd.DataFrame(bandwidth)
    for i in range(dfBandwith.shape[1]):
        dfBandwith = dfBandwith.rename(columns={i: f"Bandbreite{i+1}"})
    return dfBandwith

    

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

# Berechnet die Lautstärke des Audiosignals mit der Root-Mean-Square Methode.
# Die RMS-Methode berechnet die quadratische Mittelwertwurzel der Amplitudenwerte des Audiosignals, um die Lautstärke zu schätzen.
# Die zurückgegebenen Werte sind in einem DataFrame gespeichert, wobei jede Spalte die Lautstärke 
# für einen bestimmten Zeitabschnitt des Audiosignals darstellt.
# Die Anzahl der Spalten im DataFrame hängt von der Länge des Audiosignals und der Fensterlänge ab, die bei der Berechnung der 
# RMS-Werte verwendet wird.
def extract_loudness(file_name):
    y, sr = librosa.load(file_name)
    loudness = librosa.feature.rms(y=y)
    df_loudness = pd.DataFrame(loudness)
    for i in range(df_loudness.shape[1]):
        df_loudness = df_loudness.rename(columns={i: f"Tonstärke{i+1}"})
    return df_loudness

def plot_loudness(file_name):
    fs, data = wavfile.read(file_name)
    t = np.arange(0, len(data)/fs, 1/fs)
    fig, ax = plt.subplots()
    ax.plot(t, data)
    ax.set_xlabel('Zeit [s]')
    ax.set_ylabel('Amplitude')
    st.pyplot(fig)




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


                        
def get_features_from_single_file_df_excel(nameOfWAVFile, nameOfXLSX="default", numberOfColumns=10):
    i = 0

    df_mfcc2 = extract_mfcc(nameOfWAVFile)
    df_loudness2 = extract_loudness(nameOfWAVFile)
    df_snr2 = extract_snr(nameOfWAVFile)
    df_zcr2 = extract_zcr(nameOfWAVFile)
    df_bandwith2 = extract_bandwidth(nameOfWAVFile)
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
        mergeForth['label'] = ['']
        mergeForth.to_excel( f"{nameOfWAVFile}{nameOfXLSX}{i}" + ".xlsx")
        return f"{nameOfWAVFile}{nameOfXLSX}{i}" + ".xlsx"

        i = i+1
        


def mp4_to_wav(mp4_file, wav_file):
    audio = AudioSegment.from_file(mp4_file, format="mp4")
    audio.export(wav_file, format="wav")



def split_wav(wav_file, segment_length=3000):
    audio = AudioSegment.from_wav(wav_file)
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i+segment_length]
        segment.export(f"segment_{i//segment_length}.wav", format="wav")



def plot_mfcc(df_MFCC):
    fig, ax = plt.subplots()
    mfcc_data = np.swapaxes(df_MFCC.values, 0, 1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    ax.set_title('MFCC')
    st.pyplot(fig)
    fig, ax = plt.subplots()
    ax.plot(df_MFCC.values.T)
    ax.set_title('MFCC')
    st.pyplot(fig)

# def plot_spectrogram(file_name):
#     y, sr = librosa.load(file_name)
#     S = librosa.feature.melspectrogram(y=y, sr=sr)
#     fig, ax = plt.subplots()
#     S_dB = librosa.power_to_db(S, ref=np.max)
#     img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
#     fig.colorbar(img, ax=ax, format='%+2.0f dB')
#     ax.set(title='Mel-frequency spectrogram')
#     plt.show()


def plot_bandwidth(df):
    fig, ax = plt.subplots()
    ax.bar(df.columns, df.values[0])
    ax.set_xticklabels(df.columns, rotation=90)
    ax.set_title('Bandbreite')
    st.pyplot(fig)


def plot_zcr(df):
    fig, ax = plt.subplots()
    ax.plot(df.columns, df.values[0])
    ax.set_xticklabels(df.columns, rotation=90)
    ax.set_title('Zero Crossing Rate')
    st.pyplot(fig)

def visualize_mfcc(file_name):
    y, sr = librosa.load(file_name)
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC', xlabel='Zeit', ylabel='MFCC')
    st.pyplot(fig)


def visualize_snr(df):
    # df['id'] = range(1, len(df) + 1)
    # st.write(df)
    # i = 1

    # for i in range(df.shape[1]):
    #     i= i+1
    #     df_melted = df.melt(id_vars='id', value_vars=[f'Spektral Kontrast'+ f'{i}'], var_name='Spektral Kontrast', value_name='Wert')
    #     fig = sns.catplot(data=df_melted, x='Spektral Kontrast', y='Wert', kind='box')
    #     st.pyplot(fig)
    df_snr = df_snr.iloc[:,:10]
    ax = sns.heatmap(df_snr)
    ax.title("Spektral Kontrast")
    st.pyplot(ax.figure)


def delete_first_column_excel(file_path: str):
    wb = openpyxl.load_workbook(file_path)
    sheet = wb.active
    sheet.delete_cols(1)
    wb.save(file_path)




check = upload_and_convert()
if check == True:

    st.title("1.1 Extraktion der MFCC-Werte aus der WAV-Audio-Datei")
    st.write("MFCC (Mel Frequency Cepstral Coefficients): MFCCs werden zur automatischen Spracherkennung verwendet und führen zu einer kompakten Darstellung des Frequenzspektrums. ")

    for file in os.listdir("tempDir/"):
        if file.endswith(".wav"):
            st.write(file)
            df = extract_mfcc(f"tempDir/{file}",n_mfcc=10)
            df = df.iloc[:5, :10]
            st.caption("Ein Auszug der berechneten MFCC aus der Audiodatei dargestellt in einem Dataframe")
            st.write(df)
            st.title("1.2 Visualisierung der MFCC-Werte")
            visualize_mfcc(f"tempDir/{file}")
            st.title("1.3 Extraktion der Bandbreite einer Audioaufnahme")
            st.write("Die Bandbreite einer Audioaufnahme bezieht sich auf den Frequenzbereich, der von der Datei abgedeckt wird. Normalerweise zwischen 20 Hz - 20000 Hz.")
            df_bandwitdth = extract_bandwidth(f"tempDir/{file}")
            df_bandwitdth = df_bandwitdth.iloc[:,:30]
            st.caption("Ein Auszug der berechneten Bandbreite aus der Audiodatei dargestellt in einem Dataframe")
            st.write(df_bandwitdth)
            st.title("1.4 Visualisierung der Bandbreite einer Audioaufnahme")
            plot_bandwidth(df_bandwitdth)
            st.title("1.5 Extraktion der Zero Crossing Rate")
            st.write("Die Zero Crossing Rate (ZCR) ist eine Maßzahl für die Anzahl der Male, die ein Audiosignal die Nulllinie überquert.")
            df_zcr = extract_zcr(f"tempDir/{file}")
            df_zcr = df_zcr.iloc[:, :30]
            st.caption("Ein Auszug der berechneten Zero Crossing Rate aus der Audiodatei dargestellt in einem Dataframe")
            st.write(df_zcr)
            df_zcr = df_zcr.iloc[:,:30]
            df_zcr.columns = [f'zcr{i+1}' for i in range(len(df_zcr.columns))]
            st.title("1.6 Visualisierung der Zero Crossing Rate (ZCR)")
            plot_zcr(df_zcr)
            st.title("1.7 Extraktion des Spektral Kontrasts")
            st.write("Der Spektralkontrast einer Audioaufnahme gibt Aufschluss über die Verteilung der Energie im Frequenzspektrum der Aufnahme. Er misst den Unterschied zwischen den Spitzen und Tälern im Spektrum und kann verwendet werden, um verschiedene Eigenschaften der Aufnahme zu analysieren. Ein hoher Spektralkontrast bedeutet, dass es große Unterschiede zwischen den Spitzen und Tälern im Spektrum gibt, während ein niedriger Spektralkontrast bedeutet, dass die Energie gleichmäßiger verteilt ist.")
            df_snr = extract_snr(f"tempDir/{file}")
            df_snr = df_snr.iloc[:, :30]
            st.caption("Ein Auszug der berechneten Spektral-Kontrast-Werte aus der Audiodatei dargestellt in einem Dataframe")
            st.write(df_snr)
            st.title("1.8 Visualisierung des Spektral Kontrasts der Audiodatei")
            df_snr = df_snr.iloc[:,:10]
            ax = sns.heatmap(df_snr)
            ax.set_title("Spektral Kontrast")
            st.pyplot(ax.figure)
            st.title("1.9 Extraktion der Tonstärke der Audioaufnahme")
            st.write("Die Tonstärke der Audioaufnahme gibt an, wie laut der Ton ist. Die Tonstärke einer Audiodatei kann Aufschluss darüber geben, wie laut der Ton aufgenommen wurde und wie er sich im Vergleich zu anderen Tönen oder Geräuschen verhält. Die Tonstärke kann auch verwendet werden, um die Dynamik eines Musikstücks oder die Lautstärkeveränderungen in einer Sprachaufnahme zu analysieren. ")
            df_loudness = extract_loudness(f"tempDir/{file}")
            df_loudness = df_loudness.iloc[:,:10]
            st.caption("Ein Auszug der berechneten Tonstärke aus der Audiodatei dargestellt in einem Dataframe")
            st.write(df_loudness)
            st.title("1.10 Visualisierung der Tonstärke")
            st.write("Die Tonstärke wurde durch die Root-Mean-Square (RMS) Methode berechnet. Diese eignen sich zur Darstellung der Lautstärke für bestimmte Zeiträume.")
            plot_loudness(f"tempDir/{file}")
            excelFile = get_features_from_single_file_df_excel(f"tempDir/{file}","WAV-File")
      
            st.title("2.0 Machine Learning Modelle")
            st.title("2.1 Support Vector Machines")

            
            # delete_first_column_excel(excelFile)

            data = pd.read_excel("TrainDataForNeuronalesNetz.xlsx")
            data2 = pd.read_excel(f"{excelFile}")
            data2 = data2.drop(["Unnamed: 0"], axis=1)
            st.write(data2)
            
            data = data.dropna()
            scaler = StandardScaler()
            scaler2 = StandardScaler()
            X_data = data.drop(["label"],axis=1)
            X_data2 = data2.drop(["label"],axis=1)
            scaler = scaler.fit(data)
            scaler2 = scaler2.fit(data2)
            X_scaler_data = scaler.transform(data)
            X2_scaler2_data = scaler2.transform(data2)
            # print(X_scaler_data)
            y1 = data["label"]
            y2 = data2["label"]

            X_train, X_test, y_train, y_test = train_test_split(X_data, y1, test_size=0.2)
            model = Sequential()
            # model.add(Dense(16, input_shape=(X2.shape[1],), activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            optimizer = Adam(learning_rate=0.002)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=10)

            history = model.fit(X_train, y_train, epochs=1000, batch_size=32,
                            validation_data=(X_test, y_test), callbacks=[early_stopping])

            y_pred = model.predict(X_data2)
            y_pred = (y_pred > 0.5).astype(int)
            y_pred = pd.DataFrame(y_pred)

            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            st.write("Trainingsgenauigkeit", acc)
            st.write("Validierungsgeanuigkeit", val_acc)
            # print('Trainingsgenauigkeit:', acc)
            # print('Validierungsgenauigkeit:', val_acc)
            st.write(y_pred)



            if file.endswith(".wav"):    
                        temp_dir = tempfile.TemporaryDirectory()
                        temp_file = os.path.join(temp_dir.name, f'{excelFile}')
                        os.remove(os.path.join("tempDir", file))


    for file in os.listdir("tempDir/"):
        if file.endswith(".xlsx"):
                    print(file)
                    os.remove(os.path.join("tempDir",(file)))





