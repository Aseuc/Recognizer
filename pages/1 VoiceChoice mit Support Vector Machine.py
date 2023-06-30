import pandas as pd
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import streamlit as st
import VoiceChoice as vc
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import traceback
import ballons_red  as br
import ballons_blue as bb
import randomFacts as rf


def upload_and_convert():
    uploaded_file = st.file_uploader("Wählen Sie eine Datei zum Hochladen aus", type=["wav"], key="file_uploader6")
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
        with open(os.path.join("tempDir2",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())
        if uploaded_file.type == "video/mp4":
            # vc.mp4_to_wav(uploaded_file,vc.get_current_date_time() + ".wav")
            return True
        else:
            st.success("Hochgeladene Datei ist bereits im WAV-Format")
            return True
        
check2 = True; 
        
@st.cache_data
def load_data(excel_file):
    df = pd.read_excel(excel_file)
    return df

try:
    if upload_and_convert() == True and check2 == True:
        for file in os.listdir("tempDir2/"):
            if file.endswith(".wav"):
                vc.check_duration_uploadfile(f"tempDir2/{file}")

                excel_file = vc.get_single_excel_with_features_no_label(f"tempDir2/{file}", "tempDir2/",10,True)
                data = load_data('TrainDataNN.xlsx')
                X = data.drop(["label"],axis=1)
                y = data["label"]
                clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto'))
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                clf.fit(X, y)
                vc.delete_first_column_excel(excel_file)
                data = pd.read_excel(excel_file)
                st.write(data)
                X_data_upload = data.drop(["label"], axis=1)
                y_data_upload = data["label"]
                predictions = clf.predict(X_data_upload)
                st.write(predictions)
                # accuracy = accuracy_score(y_data_upload, y_train)
                # st.write(f'Genauigkeit: {accuracy}')

                m = 0 
                f = 0
             

                for i in predictions:
                      if i == 0: 
                            m = m+1
                      else:
                            f = f + 1

                if m > f:
                    bb.ballons_blue()
                    st.write("Person auf der Aufnahme scheint ein Mann zu sein!")
                    st.title("Wusstest du schon?: ")
                    st.write(rf.random_fact_men())
                elif f > m :
                    br.ballons_red()
                    st.write("Person auf der Aufnahme scheint eine Frau zu sein!")
                    st.title("Wusstest du schon?: ")
                    st.write(rf.random_fact_women())


                st.balloons()
                check2 = False
                os.remove(excel_file)
                
                
                
                for file in os.listdir("tempDir2/"):
                            if file.endswith(".wav"):    
                                        os.remove(os.path.join("tempDir2", file))
                            if file.endswith(".xlsx"):
                                    os.remove(os.path.join("tempDir2",(file)))
                            if file.endswith(".mp4"):
                                    os.remove(os.path.join("tempDir2",(file)))


except Exception as e:
        tb = traceback.format_exc()
        print(f"Ein Fehler ist aufgetreten:\n{tb}")