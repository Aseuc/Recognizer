from audio_recorder_streamlit import audio_recorder
import VoiceChoice as vc
import streamlit as st
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import pandas as pd
import ballons_red as br
import ballons_blue as bb
import randomFacts
import base64

st.set_page_config(
    layout="wide",
    page_icon="favicon.ico",
    page_title="VoiceChoice - NN & Echzeitaufnahme",
    initial_sidebar_state="auto"

)

vc.add_logo_sidebar()


def neuronal_network(excel_file_train_data, excel_file_test_data, layers=0, neurons=0):
    vc.delete_first_column_excel(excel_file_test_data)
    # add_id_column(excelFile)
    data = pd.read_excel(excel_file_train_data)
    data2 = pd.read_excel(excel_file_test_data)
    # data2 = data2.drop(["Unnamed: 0"], axis=1)
    st.write(data2)

    data = data.dropna()
    scaler = StandardScaler()
    # scaler2 = StandardScaler()
    X_data = data.drop(["label"], axis=1)
    X_data2 = data2.drop(["label"], axis=1)
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
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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
    button_style = """
          <style>
          .custom-button.button1 {
              background-color: #0000FF; /* Blau */
              color: #FFFFFF;
              border-color: #0000FF;
              border-radius: 5px;
              padding: 0.5rem 1rem;
          }

          .custom-button.button2 {
              background-color: #FF00FF; /* Pink */
              color: #FFFFFF;
              border-color: #FF00FF;
              border-radius: 5px;
              padding: 0.5rem 1rem;
          }

          .custom-button.button3 {
              background-color: #FFA500; /* Orange */
              color: #FFFFFF;
              border-color: #FFA500;
              border-radius: 5px;
              padding: 0.5rem 1rem;
          }
           .custom-button.button4 {
          background-color: #00FF00; /* Grün */
          color: #FFFFFF;
          border-color: #00FF00;
          border-radius: 5px;
          padding: 0.5rem 1rem;
      }
      </style>
      """

    # CSS-Stil in Streamlit einfügen
    st.markdown(button_style, unsafe_allow_html=True)
    val_acc = val_acc[len(val_acc) - 1]
    acc = acc[len(acc) - 1]
    # Button 1 - Analyse von Essen
    if st.button("Trainingsgenauigkeit", key="button1"):
        st.write("Trainingsgenauigkeit", acc, unsafe_allow_html=True)

    if st.button("Validierungsgauigkeit", key="button2"):
        st.write("Validierungsgenauigkeit", val_acc, unsafe_allow_html=True)

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
            "<h3 style='text-align: center;'>Auf der gesprochenen Audioaufnahme spricht wahrscheinlich ein "
            "Mann!</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<h2 style='text-align: center;'>Wusstest du schon?</h2>", unsafe_allow_html=True)
        st.markdown(
            f"<h2 style='text-align: center;'>{randomFacts.random_fact_men()}</h2>", unsafe_allow_html=True)
    elif countOne > countZero:
        br.ballons_red()
        st.markdown(
            "<h3 style='text-align: center;'>Auf der gesprochenen Audioaufnahme spricht wahrscheinlich eine"
            "Frau!</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<h3 style='text-align: center;'>Wusstest du schon?</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<h3 style='text-align: center;'>{randomFacts.random_fact_women()}</h3>", unsafe_allow_html=True)
    return


audio_bytes = audio_recorder("Mikrofon anklicken um Aufnahme zu starten!")

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    with open("tempDir2/record.wav", "wb") as f:
        f.write(audio_bytes)

if st.button("Neuronales Netz Klassifizierung starten!"):
    excel_file = vc.get_single_excel_with_features_no_label("tempDir2/record.wav", "tempDir2/", 1, False)

    vc.neuronal_network("TDNN3Z.xlsx", excel_file,5, [32,32,32,32,32])
    os.remove(excel_file)
    os.remove("tempDir2/record.wav")
