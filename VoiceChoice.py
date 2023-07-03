import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy.io import wavfile
import librosa
import wave
import random
from pydub import AudioSegment
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from datetime import datetime
import librosa.display
import openpyxl
from PIL import Image
import ballons_red as br
import ballons_blue as bb
import randomFacts
import base64


# Fügt der Sidebar unser Logo hinzu
def add_logo_sidebar():
    with open("voicechoicelogo.png", "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    st.sidebar.markdown(
        f"""
        <div>
            <img style='width=1px; height=2px;' src="data:image/png;base64,{data}">
        </div>
        """,
        unsafe_allow_html=True,
    )


# Gibt die Dauer der Audiodatei aus
def get_duration(audio_file_path):
    audio, sr = librosa.load(audio_file_path)
    duration = librosa.get_duration(y=audio, sr=sr)
    return duration


# Wird benötigt um zu schauen, die Audiodatei größer als 3 Sekunden groß ist, wenn ja muss diese in 3 Sekunden Sequenzen gesplittet werden zur Verarbeitung
def check_duration_uploadfile(uploadfile_path, allowed_duration=3):
    duration = get_duration(uploadfile_path)
    output_file = None

    if duration > 3:
        output_file = extract_random_sequence(uploadfile_path, uploadfile_path, 3)
        st.write(f"Dauer der WAV-Datei beträgt: {duration} Sekunden!")
    else:
        st.write(f"Dauer der WAV-Datei beträgt: {duration} Sekunden!")
    return output_file


# Extrahiert die Zero Crossing Rate aus der Audiodatei
def extract_zcr(file_name):
    y, sr = librosa.load(file_name)
    zcr = librosa.feature.zero_crossing_rate(y)
    df_ZCR = pd.DataFrame(zcr)
    for i in range(df_ZCR.shape[1]):
        df_ZCR = df_ZCR.rename(columns={i: f"Zero Crossing Rate{i + 1}"})
    return df_ZCR


# Extrahiert den Spektral Kontrast aus der Audiodatei
def extract_snr(file_name):
    y, sr = librosa.load(file_name)
    S = np.abs(librosa.stft(y))
    snr = librosa.feature.spectral_contrast(S=S)
    dfsnr = pd.DataFrame(snr)
    for i in range(dfsnr.shape[1]):
        dfsnr = dfsnr.rename(columns={i: f"Spektral Kontrast{i + 1}"})
    return dfsnr


# Extrahiert die Bandbreite aus der Audiodatei
def extract_bandwidth(file_name):
    y, sr = librosa.load(file_name)
    S = np.abs(librosa.stft(y))
    bandwidth = librosa.feature.spectral_bandwidth(S=S)
    dfBandwith = pd.DataFrame(bandwidth)
    for i in range(dfBandwith.shape[1]):
        dfBandwith = dfBandwith.rename(columns={i: f"Bandbreite{i + 1}"})
    return dfBandwith


# Holt sich das aktuelle Datum und Uhrzeit auf Millisekunden genau, wichtig zur
# Setzung eines Markers, zur Unterscheidung von Dateien die im Programmverlauf erstellt und verwendet werden.
def get_current_date_time():
    now = datetime.now()
    date_time_str = now.strftime("%d_%m_%Y_%H_%M_%S_%f_%Z")
    return date_time_str + "_"


# Splittet die Audiodatei in 3 Sekunden Sequenzen auf falls nichts anderes übergeben wird.
def split_wav(destinationPath, wav_file, segment_length=3000):
    audio = AudioSegment.from_wav(wav_file)

    for i in range(0, len(audio), segment_length):
        segment = audio[i:i + segment_length]
        segment.export(f"{destinationPath}segment__{i - 10 // segment_length}.wav", format="wav")


# Wird nicht mehr benötigt
def get_n_frames_duration(input_file_path, duration):
    with wave.open(input_file_path, 'rb') as input_wav:
        n_channels = input_wav.getnchannels()
        sample_width = input_wav.getsampwidth()
        frame_rate = input_wav.getframerate()
        n_frames = input_wav.getnframes()

        n_frames_duration = int(frame_rate * duration)
        return n_frames_duration

    # Falls diese Funktion und die split_multiple_frames() nicht mehr funktioniert lösche nameOfXLSX


# Extrahiert eine Zufällige Sequence aus der Audiodatei
def extract_random_sequence(input_file_path, output_file_path, duration=3):
    with wave.open(input_file_path, 'rb') as input_wav:
        n_channels = input_wav.getnchannels()
        sample_width = input_wav.getsampwidth()
        frame_rate = input_wav.getframerate()
        n_frames = input_wav.getnframes()

        n_frames_duration = int(frame_rate * duration)

        if n_frames_duration > n_frames:
            raise ValueError(
                f"Die angegebene Dauer ({duration} Sekunden) ist länger als die Gesamtdauer der Datei ({n_frames / frame_rate} Sekunden).")

        start_frame = random.randint(0, n_frames - n_frames_duration)

        input_wav.setpos(start_frame)
        frames = input_wav.readframes(n_frames_duration)

    with wave.open(f"{output_file_path}", 'wb') as output_wav:
        output_wav.setparams((n_channels, sample_width, frame_rate, n_frames_duration, 'NONE', 'not compressed'))
        output_wav.writeframes(frames)
    return output_file_path


# Extrahiert die Frames aus der Audiodatei
def get_n_frames(input_file_path):
    with wave.open(input_file_path, 'rb') as input_wav:
        n_channels = input_wav.getnchannels()
        sample_width = input_wav.getsampwidth()
        frame_rate = input_wav.getframerate()
        n_frames = input_wav.getnframes()

        # n_frames_duration = int(frame_rate * duration)
        return n_frames


# Splittet mehre Audiodatei aufeinmal in 3 Sekunden Sequenzen, wurde beim Data Preprocessing verwendet.
def split_multiple_frames(input_file_path, output_file_path, duration=3):
    i = 0
    for files in os.listdir(input_file_path):

        if files.endswith(".wav"):
            # if get_n_frames_duration("longMenAudios/" + files,duration=3) > get_n_frames("longMenAudios/" + files):
            extract_random_sequence(input_file_path + files,
                                    output_file_path + f"{get_current_date_time()}duration_{duration}", duration=3)

            print(f"Random{i}duration_{duration}")
            i = i + 1


# Funktion um bestimmte Daten in Ordnern schnell umzubennen.
def rename_data(file_path="MenSequences" or "WomenSequences", files_to_rename="Random"):
    i = 0
    for files in os.listdir(file_path):
        if files.endswith(".wav"):
            new_name = f"{files_to_rename}_{i}.wav"
            os.rename(os.path.join(file_path, files), os.path.join(file_path, new_name))
            i += 1


# Uploadfunktion um Audiodatei in Streamlit hochzuladen
def upload_and_convert():
    uploaded_file = st.file_uploader("Wählen Sie eine Datei zum Hochladen aus", type=["mp4", "wav"],
                                     key="file_uploader")
    print(uploaded_file)
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        st.write(file_details)
        with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        if uploaded_file.type == "video/mp4":
            mp4_to_wav(uploaded_file, get_current_date_time() + ".wav")
            return True
        else:
            st.success("Hochgeladene Datei ist bereits im WAV-Format")
            return True


# Extrahiert alle Features aus der Audiodateien in einem Ordner auf einen Schlag und speichert diese als CSV
def get_features_df_csv(nameOfCsv, ordner_path):
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
            df_loudness2 = pd.DataFrame(index=[1], columns=['Tonstärke'])
            df_snr2 = pd.DataFrame(index=[2], columns=['Spektral Kontrast'])
            df_zcr2 = pd.DataFrame(index=[3], columns=['Zero Crossing Rate'])
            df_mfcc2 = pd.DataFrame(index=[4], columns=['MFCC'])

            df_bandwith2.at[0, 'Bandbreite'] = bandwidth_vec2.tolist()
            df_loudness2.at[1, 'Tonstärke'] = chroma_vec2.tolist()
            df_snr2.at[2, 'Spektral Kontrast'] = sc_vec2.tolist()
            df_zcr2.at[3, 'Zero Crossing Rate'] = zcr_vec2.tolist()
            df_mfcc2.at[4, 'MFCC'] = mfcc_vec2.tolist()

            df_bandwith2['id'] = range(1, len(df_bandwith2) + 1)
            df_loudness2['id'] = range(1, len(df_loudness2) + 1)
            df_snr2['id'] = range(1, len(df_snr2) + 1)
            df_zcr2['id'] = range(1, len(df_zcr2) + 1)
            df_mfcc2['id'] = range(1, len(df_mfcc2) + 1)

            mergeFirst = pd.merge(df_mfcc2, df_zcr2, on='id')
            mergeSecond = pd.merge(mergeFirst, df_loudness2, on='id')
            mergeThird = pd.merge(mergeSecond, df_snr2, on='id')
            mergeForth = pd.merge(mergeThird, df_bandwith2, on='id')
            mergeForth['Frau/Mann'] = ['Frau']
            mergeForth.to_csv(f"{nameOfCsv}{i}.csv")

            i += 1


# Extrahiert aus der Audiodatei die MFCC-Werte
def extract_mfcc(file_name, n_mfcc=13):
    y, sr = librosa.load(file_name)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    df_MFCC = pd.DataFrame(mfcc)
    for i in range(df_MFCC.shape[1]):
        df_MFCC = df_MFCC.rename(columns={i: f"MFCC{i + 1}"})
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
        df_loudness = df_loudness.rename(columns={i: f"Tonstärke{i + 1}"})
    return df_loudness


# Wurde mal zum Plotten der Tonstärke verwendet.
# Problem war/ist, die Plotts wurden durch das setzen des Seitenlayouts "st.set_page_config()" auf "wide" zu groß angezeigt.
# Gilt für alle anderen Plot Funktionen
def plot_loudness(file_name):
    fs, data = wavfile.read(file_name)
    t = np.arange(0, len(data) / fs, 1 / fs)
    fig, ax = plt.subplots()
    ax.plot(t, data)
    ax.set_xlabel('Zeit [s]')
    ax.set_ylabel('Amplitude')
    st.pyplot(fig)


# Extrahiert alle Features aus der Audiodateien in einem Ordner auf einen Schlag und speichert diese als XLSX

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

            df_bandwith2['id'] = range(1, len(df_bandwith2) + 1)
            df_loudness2['id'] = range(1, len(df_loudness2) + 1)
            df_snr2['id'] = range(1, len(df_snr2) + 1)
            df_zcr2['id'] = range(1, len(df_zcr2) + 1)
            df_mfcc2['id'] = range(1, len(df_mfcc2) + 1)

            mergeFirst = pd.merge(df_mfcc2, df_zcr2, on='id')
            mergeSecond = pd.merge(mergeFirst, df_loudness2, on='id')
            mergeThird = pd.merge(mergeSecond, df_snr2, on='id')
            mergeForth = pd.merge(mergeThird, df_bandwith2, on='id')
            mergeForth.dropna()

            num_rows = mergeForth.shape[0]

            for i in range(num_rows):
                mergeForth.at[i, 'id'] = i
                i = i + 1
                mergeForth = mergeForth.rename(columns={'Unnamed: 0': 'ID'})
                mergeForth.dropna()
                mergeForth = mergeForth.drop("id", axis=1)
                # mergeForth.head()
                # mergeForth = mergeForth.sample(frac=1).reset_index(drop=True)
                if labelType == "Frau" or labelType == "Mann" or labelType == 1 or labelType == 0:
                    mergeForth['label'] = [f'{labelType}']
                    mergeForth.to_excel(f"{destinationPath}{nameOfXLSX}{i}.xlsx")

        i = i + 1


# Extrahiert alle Features aus der Audiodatei auf einen Schlag und speichert diese als XLSX aber nur eine Datei

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
    df_bandwith2['id'] = range(1, len(df_bandwith2) + 1)
    df_loudness2['id'] = range(1, len(df_loudness2) + 1)
    df_snr2['id'] = range(1, len(df_snr2) + 1)
    df_zcr2['id'] = range(1, len(df_zcr2) + 1)
    df_mfcc2['id'] = range(1, len(df_mfcc2) + 1)
    mergeFirst = pd.merge(df_mfcc2, df_zcr2, on='id')
    mergeSecond = pd.merge(mergeFirst, df_loudness2, on='id')
    mergeThird = pd.merge(mergeSecond, df_snr2, on='id')
    mergeForth = pd.merge(mergeThird, df_bandwith2, on='id')
    mergeForth.dropna()
    num_rows = mergeForth.shape[0]
    for i in range(num_rows):
        mergeForth.at[i, 'id'] = i
        i = i + 1
        mergeForth = mergeForth.rename(columns={'Unnamed: 0': 'ID'})
        mergeForth.dropna()
        mergeForth = mergeForth.drop("id", axis=1)
        # mergeForth.head()
        # mergeForth = mergeForth.sample(frac=1).reset_index(drop=True)
        mergeForth['label'] = ['']
        mergeForth.to_excel(f"{nameOfWAVFile}{nameOfXLSX}{i}" + ".xlsx")
        return f"{nameOfWAVFile}{nameOfXLSX}{i}" + ".xlsx"

        i = i + 1


# Konvertiert mp4 zu wav Dateien
def mp4_to_wav(mp4_file, wav_file):
    audio = AudioSegment.from_file(mp4_file, format="mp4")
    audio.export(f"tempDir/{wav_file}", format="wav")


# Redundate Funktion wird beibehalten falls, dennoch benötigt
def split_wav(wav_file, segment_length=3000):
    audio = AudioSegment.from_wav(wav_file)
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i + segment_length]
        segment.export(f"segment_{i // segment_length}.wav", format="wav")


#
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
    st.markdown(f"<div >{st.pyplot(fig)}</div>")


def visualize_snr(df):
    df_snr = df_snr.iloc[:, :10]
    ax = sns.heatmap(df_snr)
    ax.title("Spektral Kontrast")
    st.pyplot(ax.figure)


# Entfernt erste Spalte der Excel
# Arbeitsmappe, zur Normalisierung der Spalten. Sinn: Anzahl der Spalten von Trainingsdatensatz und Predictiondatensatz müssen gleich sein.
def delete_first_column_excel(file_path: str):
    wb = openpyxl.load_workbook(file_path)
    sheet = wb.active
    sheet.delete_cols(1)
    wb.save(file_path)


# Diese Funktion ist wichtig, da Sie einzeilige Werte wie ZCR, Tonstärke auf die Mehrzeiligen Werte verteilt
def duplicate_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.loc[df.index.repeat(n)].reset_index(drop=True)
    return df


# Wichtig zur Extraktion von Features aus der Audiodatei, aber ohne diese zu Labeln.
# Vorherige Funktionen sind dazu bestimmt die xlsx,csv und dessen Inhalt zu Labeln
def get_single_excel_with_features_no_label(inputfile_path, outputfile_path, features_size, duplicatesRows=True):
    file = inputfile_path
    df_mfcc = extract_mfcc(file, features_size)
    df_mfcc.dropna()
    df_mfcc = df_mfcc.iloc[:5, :features_size]

    df_snr = extract_snr(file)
    df_snr.dropna()
    df_snr = df_snr.iloc[:5, :features_size]

    df_loudness = extract_loudness(file)
    df_loudness.dropna()
    df_loudness = df_loudness.iloc[:, :features_size]
    if duplicatesRows == True:
        df_loudness = duplicate_rows(df_loudness, 5)

    df_bandwith = extract_bandwidth(file)
    df_bandwith.dropna()
    df_bandwith = df_bandwith.iloc[:, :features_size]
    df_bandwith.head()
    if duplicatesRows == True:
        df_bandwith = duplicate_rows(df_bandwith, 5)

    df_zcr = extract_zcr(file)
    df_zcr.dropna()
    df_zcr = df_zcr.iloc[:, :features_size]
    df_zcr.head()
    if duplicatesRows == True:
        df_zcr = duplicate_rows(df_zcr, 5)

    df_bandwith['id'] = range(1, len(df_bandwith) + 1)
    df_loudness['id'] = range(1, len(df_loudness) + 1)
    df_snr['id'] = range(1, len(df_snr) + 1)
    df_zcr['id'] = range(1, len(df_zcr) + 1)
    df_mfcc['id'] = range(1, len(df_mfcc) + 1)

    marker = get_current_date_time()
    mergeFirst = pd.merge(df_mfcc, df_snr, on='id')
    mergeSecond = pd.merge(mergeFirst, df_loudness, on='id')
    mergeThird = pd.merge(mergeSecond, df_bandwith, on="id")
    mergeForth = pd.merge(mergeThird, df_zcr, on="id")
    mergeForth = mergeForth.drop(["id"], axis=1)
    if (duplicatesRows == False):
        mergeForth['label'] = [""]
    elif (duplicatesRows == True):
        mergeForth["label"] = ["", "", "", "", ""]

    mergeForth.to_excel(f"{outputfile_path}{marker}" + ".xlsx")

    return outputfile_path + marker + ".xlsx"


# Redundant wird beibehalten falls dennoch benötigt
def create_model(layers, neurons):
    model = Sequential()
    for i in range(layers):
        model.add(Dense(neurons[i], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


# Baut ein individuelles Neuronales Netz auf
def neuronal_network(excel_file_train_data, excel_file_test_data, layers=0, neurons=0):
    delete_first_column_excel(excel_file_test_data)
    # add_id_column(excelFile)
    data = pd.read_excel(excel_file_train_data)
    data2 = pd.read_excel(excel_file_test_data)
    # data2 = data2.drop(["Unnamed: 0"], axis=1)

    data = data.dropna()
    scaler = StandardScaler()
    # scaler2 = StandardScaler()
    X_data = data.drop(["label", "MFCC1", "Spektral Kontrast1", "Tonstärke1"], axis=1)
    X_data2 = data2.drop(["label", "MFCC1", "Spektral Kontrast1", "Tonstärke1"], axis=1)
    scaler = scaler.fit(X_data)
    X_scaler_data = scaler.transform(X_data)
    X2_scaler2_data = scaler.transform(X_data2)
    # st.write(X_scaler_data)
    # st.write(X2_scaler2_data)
    y1 = data["label"]
    y2 = data2["label"]

    X_train, X_test, y_train, y_test = train_test_split(X_scaler_data, y1, test_size=0.2)
    model = Sequential()
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(8, activation='relu'))

    for i in range(layers):
        model.add(Dense(neurons[i], activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.002)
    model.compile(loss='binary_cross entropy', optimizer=optimizer, metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(X_train, y_train, epochs=1000, batch_size=32,
                        validation_data=(X_test, y_test), callbacks=[early_stopping])

    # with open('model.pkl', 'wb') as file:
    #     pickle.dump(model, file)

    # with open('model1.pkl', 'rb') as file:
    # model = pickle.load(file)
    # model.save('my_model.h5')
    y_pred = model.predict(X2_scaler2_data)
    # y_pred2 = model.predict(X_data2)
    y_pred = (y_pred > 0.5).astype(int)
    # y_pred = pd.DataFrame(y_pred)

    # countZero = 0
    # countOne = 0
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    st.write("Trainingsgenauigkeit", acc)
    st.write("Validierungsgenauigkeit", val_acc)
    st.write(y_pred)
    countZero = 0
    countOne = 0
    for i in y_pred:
        if (i == 0):
            countZero = countZero + 1
        else:
            countOne = countOne + 1

    if countZero > countOne:
        bb.ballons_blue()
        st.markdown(
            "<h3 style='text-align: center;'>Auf der gesprochenen Audiodatei spricht wahrscheinlich ein "
            "Mann!</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<h2 style='text-align: center;'>Wusstest du schon?</h2>", unsafe_allow_html=True)
        st.markdown(
            f"<h2 style='text-align: center;'>{randomFacts.random_fact_men()}</h2>", unsafe_allow_html=True)
    elif countOne > countZero:
        br.ballons_red()
        st.markdown(
            "<h3 style='text-align: center;'>Auf der gesprochenen Audiodatei spricht wahrscheinlich eine "
            "Frau!</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<h3 style='text-align: center;'>Wusstest du schon?</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<h3 style='text-align: center;'>{randomFacts.random_fact_women()}</h3>", unsafe_allow_html=True)
    return


# Fügt eine Spalte einer Excel-Datei hinzu
def add_id_column(excel_file: str):
    df = pd.read_excel(excel_file)
    df.insert(0, 'ID', range(1, len(df) + 1))
    df.to_excel(excel_file, index=False)


# check = upload_and_convert()
# file_name = None
# check2 = True
# if check == True and check2 == True:
#     for file in os.listdir("tempDir/"):
#         if file.endswith(".wav"):
#             with st.expander("1.1 Extraktion der MFCC-Werte aus der WAV-Audio-Datei"):      
#                 st.title("1.1 Extraktion der MFCC-Werte aus der WAV-Audio-Datei")
#                 st.write("MFCC (Mel Frequency Cepstral Coefficients): MFCCs werden zur automatischen Spracherkennung verwendet und führen zu einer kompakten Darstellung des Frequenzspektrums. ")
#                 df = extract_mfcc(f"tempDir/{file}",n_mfcc=10)
#                 df = df.iloc[:5, :10]
#                 st.caption("Ein Auszug der berechneten MFCC aus der Audiodatei dargestellt in einem Dataframe")
#                 st.write(df)

#             with st.expander("1.2 Visualisierung der MFCC-Werte"):      
#                 st.title("1.2 Visualisierung der MFCC-Werte")
#                 visualize_mfcc(f"tempDir/{file}")


#             with st.expander("1.3 Extraktion der Bandbreite einer Audioaufnahme"): 
#                 st.title("1.3 Extraktion der Bandbreite einer Audioaufnahme")
#                 st.write("Die Bandbreite einer Audioaufnahme bezieht sich auf den Frequenzbereich, der von der Datei abgedeckt wird. Normalerweise zwischen 20 Hz - 20000 Hz.")
#                 df_bandwitdth = extract_bandwidth(f"tempDir/{file}")
#                 df_bandwitdth = df_bandwitdth.iloc[:,:30]
#                 st.caption("Ein Auszug der berechneten Bandbreite aus der Audiodatei dargestellt in einem Dataframe")
#                 st.write(df_bandwitdth)

#             with st.expander("1.4 Visualisierung der Bandbreite einer Audioaufnahme"):
#                 st.title("1.4 Visualisierung der Bandbreite einer Audioaufnahme")
#                 plot_bandwidth(df_bandwitdth)

#             with st.expander("1.5 Extraktion der Zero Crossing Rate"):
#                 st.title("1.5 Extraktion der Zero Crossing Rate")   
#                 st.write("Die Zero Crossing Rate (ZCR) ist eine Maßzahl für die Anzahl der Male, die ein Audiosignal die Nulllinie überquert.")
#                 df_zcr = extract_zcr(f"tempDir/{file}")
#                 df_zcr = df_zcr.iloc[:, :30]
#                 st.caption("Ein Auszug der berechneten Zero Crossing Rate aus der Audiodatei dargestellt in einem Dataframe")
#                 st.write(df_zcr)
#                 df_zcr = df_zcr.iloc[:,:30]
#                 df_zcr.columns = [f'zcr{i+1}' for i in range(len(df_zcr.columns))]

#             with st.expander("1.6 Visualisierung der Zero Crossing Rate (ZCR)"):
#                 st.title("1.6 Visualisierung der Zero Crossing Rate (ZCR)")
#                 plot_zcr(df_zcr)

#             with st.expander("1.7 Extraktion des Spektral Kontrasts"):
#                 st.title("1.7 Extraktion des Spektral Kontrasts")
#                 st.write("Der Spektralkontrast einer Audioaufnahme gibt Aufschluss über die Verteilung der Energie im Frequenzspektrum der Aufnahme. Er misst den Unterschied zwischen den Spitzen und Tälern im Spektrum und kann verwendet werden, um verschiedene Eigenschaften der Aufnahme zu analysieren. Ein hoher Spektralkontrast bedeutet, dass es große Unterschiede zwischen den Spitzen und Tälern im Spektrum gibt, während ein niedriger Spektralkontrast bedeutet, dass die Energie gleichmäßiger verteilt ist.")
#                 df_snr = extract_snr(f"tempDir/{file}")
#                 df_snr = df_snr.iloc[:, :30]
#                 st.caption("Ein Auszug der berechneten Spektral-Kontrast-Werte aus der Audiodatei dargestellt in einem Dataframe")
#                 st.write(df_snr)

#             with st.expander("1.8 Visualisierung des Spektral Kontrasts der Audiodatei"):
#                 st.title("1.8 Visualisierung des Spektral Kontrasts der Audiodatei")
#                 df_snr = df_snr.iloc[:,:10]
#                 ax = sns.heatmap(df_snr)
#                 ax.set_title("Spektral Kontrast")
#                 st.pyplot(ax.figure)


#             with st.expander("1.9 Extraktion der Tonstärke der Audioaufnahme"):
#                 st.title("1.9 Extraktion der Tonstärke der Audioaufnahme")
#                 st.write("Die Tonstärke der Audioaufnahme gibt an, wie laut der Ton ist. Die Tonstärke einer Audiodatei kann Aufschluss darüber geben, wie laut der Ton aufgenommen wurde und wie er sich im Vergleich zu anderen Tönen oder Geräuschen verhält. Die Tonstärke kann auch verwendet werden, um die Dynamik eines Musikstücks oder die Lautstärkeveränderungen in einer Sprachaufnahme zu analysieren. ")
#                 df_loudness = extract_loudness(f"tempDir/{file}")
#                 df_loudness = df_loudness.iloc[:,:10]
#                 st.caption("Ein Auszug der berechneten Tonstärke aus der Audiodatei dargestellt in einem Dataframe")
#                 st.write(df_loudness)

#             with st.expander("1.10 Visualisierung der Tonstärke"):

#                 st.title("1.10 Visualisierung der Tonstärke")
#                 st.write("Die Tonstärke wurde durch die Root-Mean-Square (RMS) Methode berechnet. Diese eignen sich zur Darstellung der Lautstärke für bestimmte Zeiträume.")
#                 plot_loudness(f"tempDir/{file}")


#             with st.expander("2.0 Machine Learning Modelle"):
#                 st.title("2.0 Machine Learning Modelle")
#                 st.title("2.1 Neuronales Netzwerk")
#                 st.write("2.1.1 Das Neuronale Netz berechnet hier, ob die hochgeladene Audioaufnahme von einem Mann oder einer Frau gesprochen wurde. Hier wird jedoch nur eine Zeile der Sequence verwendet siehe Dataframe:")
#                 excelFile = get_single_excel_with_features_no_label(f"tempDir/{file}","tempDir/",10,False)
#                 neuronal_network("TrainDataForNeuronalesNetz (1).xlsx",excelFile)
#                 # st.write("2.1.2 Das Neuronale Netz berechnet hier, ob die hochgeladene Audioaufnahme von einem Mann oder einer Frau gesprochen wurde. Hier wird jedoch ein Block aus 5 Zeilen der Sequence verwendet siehe Dataframe und das Neuronale Netz besteht aus einer Schicht mit der Aktivierungsfuntion " "Sigmoid" + ":")

#                 # excelFile = get_single_excel_with_features_no_label(f"tempDir/{file}","tempDir/",10,True)
#                 # neuronal_network("TrainDataFuerNeuronalesNetzohneGroupID.xlsx",excelFile,0,0)


#                 # st.write("2.1.3 Im vergleich zu 2.1.2 ändert sich nun die Anzahl der Schichten von 1 auf 2 Schichten. Dabei wird in der ersten Schicht 8 Neuronen verwendet und in der zweiten Schicht 16 Neuronen verwendet.")
#                 # excelFile = get_single_excel_with_features_no_label(f"tempDir/{file}","tempDir/",10,True)
#                 # neuronal_network("TrainDataFuerNeuronalesNetzohneGroupID.xlsx",excelFile,2,[8,16])


#                 file_name = file
#                 check2 == False
#                 for file in os.listdir("tempDir/"):
#                             if file.endswith(".wav"):    
#                                         temp_dir = tempfile.TemporaryDirectory()
#                                         temp_file = os.path.join(temp_dir.name, f'{excelFile}')
#                                         os.remove(os.path.join("tempDir", file))
#                             if file.endswith(".xlsx"):
#                                     print(file)
#                                     os.remove(os.path.join("tempDir",(file)))
#                             if file.endswith(".mp4"):
#                                     os.remove(os.path.join("tempDir",(file)))

st.set_page_config(
    page_title="VoiceChoice",
    page_icon="favicon.ico",
    layout='wide',
    initial_sidebar_state="collapsed"
)

col1, col2, col3 = st.columns(3)
with col2:
    image = Image.open('vc.png')
    st.image(image)

st.markdown("<h1 style='text-align: center;'>Herzlich Willkomnen zu VoiceChoice!</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Your voice is your choice!</h3>", unsafe_allow_html=True)

st.markdown(
    "<h3 style='text-align: center;'>Mit VoiceChoice hast du die Möglichkeit mit verschiedensten Machine Learning Modellen das Geschlecht einer Person auf der Audioaufnahme klassifizieren.</h3>",
    unsafe_allow_html=True)

st.header("1. VoiceChoice mit Neuronalen Netzwerk")
st.write(
    "Funktionsweise: Es ist ganz einfach! Du die Möglichkeit per .wav Datei mit einem neuronalen Netzwerk die Person zu klassifizieren. Mit Browse File lädst du die Datei hoch ""(siehe Bild)" "und schon startet der Prozess der Klassifizierung!")
image = Image.open('UploadPic.png')
st.image("UploadPic.png")
