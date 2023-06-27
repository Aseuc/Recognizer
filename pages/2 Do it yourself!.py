
# import streamlit as st
# import os
# import VoiceChoice as vc
# from keras.models import Sequential
# from keras.layers  import Dense, LSTM, Dropout
# from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
# def upload_and_convert_newPath(new_path):
#     uploaded_file2 = st.file_uploader("Wählen Sie eine Datei zum Hochladen aus", type=["mp4", "wav"], key="file_uploader2")
#     if uploaded_file2 is not None:
#         file_details = {"FileName":uploaded_file2.name,"FileType":uploaded_file2.type}
#         st.write(file_details)
#         with open(os.path.join(new_path,uploaded_file2.name),"wb") as f:
#             f.write(uploaded_file2.getbuffer())
#         if uploaded_file2.type == "video/mp4":
#             vc.mp4_to_wav(uploaded_file2,vc.get_current_date_time() + ".wav")
#             return True
#         else:
#             st.success("Hochgeladene Datei ist bereits im WAV-Format")
#             return True



# if upload_and_convert_newPath("tempDir2") == True:

#     max_layers = 5
#     layer_options = list(range(1, max_layers + 1))

#     if 'num_layers' not in st.session_state:
#         st.session_state.num_layers = layer_options[0]

#     num_layers = st.selectbox("Select number of layers", layer_options, key='num_layers')

#     neurons = []
#     for i in range(num_layers):
#         if f'neurons_{i}' not in st.session_state:
#             st.session_state[f'neurons_{i}'] = 16
#         n = st.number_input(f"Select number of neurons in layer {i + 1}", min_value=1, value=st.session_state[f'neurons_{i}'], key=f'neurons_{i}')
#         neurons.append(n)

#     st.write(f"Number of layers: {num_layers}")
#     st.write(f"Number of neurons in each layer: {neurons}")

#     if st.button('Start'):
#             excelFile = vc.get_single_excel_with_features_no_label(f"tempDir2/","tempDir2/",10,True)
#             vc.neuronal_network("TrainDataFuerNeuronalesNetzohneGroupID",excelFile,num_layers,neurons)

