import pandas as pd
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import streamlit as st
import VoiceChoice as vc
import os
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




def upload_and_convert():
    uploaded_file = st.file_uploader("Wählen Sie eine Datei zum Hochladen aus", type=["mp4", "wav"], key="file_uploader4")
    print(uploaded_file)
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
        st.write(file_details)
        with open(os.path.join("tempDir2",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())
        if uploaded_file.type == "video/mp4":
            vc.mp4_to_wav(uploaded_file,vc.get_current_date_time() + ".wav")
            return True
        else:
            st.success("Hochgeladene Datei ist bereits im WAV-Format")
            return True

if upload_and_convert() == True:
    for file in os.listdir("tempDir2/"):
        if file.endswith(".wav"):
            # svc = SVC(kernel='linear', C=1)
            
            excel_file = vc.get_single_excel_with_features_no_label(f"tempDir2/{file}", "tempDir2/",1,False)
            
            data = pd.read_excel('TrainDataRFohneID1.xlsx')
            
            X = data.drop(["label"],axis=1)
            
            y = data["label"]
            
            clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto'))
            
            clf.fit(X, y)
            
            vc.delete_first_column_excel(excel_file)
            data = pd.read_excel(excel_file)
            st.write(data)
            X = data.drop(["label"], axis=1)
            y = data["label"]
            predictions = clf.predict(X)
            st.write(predictions)
            
            # Laden Sie das Iris-Datensatz
            iris = datasets.load_iris()
            X = data.drop(["label"], axis=1)
            y = data["label"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            st.write(X_train)
            svc = SVC(kernel='linear', C=1)
            svc.fit(X_train, y_train)
            y_pred = svc.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f'Genauigkeit: {accuracy}')


            if predictions == 0: 
                st.write("Person auf der Aufnahme scheint ein Mann zu sein!")
            else:
                st.write("Person auf der Aufnahme scheint eine Frau zu sein!")
            for file in os.listdir("tempDir2/"):
                        if file.endswith(".wav"):    
                                    os.remove(os.path.join("tempDir2", file))
                        if file.endswith(".xlsx"):
                                os.remove(os.path.join("tempDir2",(file)))
                        if file.endswith(".mp4"):
                                os.remove(os.path.join("tempDir2",(file)))