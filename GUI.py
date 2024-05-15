import streamlit as st
import pandas as pd
import numpy as np
from Proses import preprocess_data,split_and_tfidf,svm_cs_results

st.title("""KLASIFIKASI JUDUL BERITA BAHASA INDONESIA """)

def upload_dataset():
    upload_file = st.file_uploader("Upload Dataset dalam bentuk xlsx", type=["xlsx"])
    if upload_file is not None:
        #dataset = pd.read_excel(upload_file, sep=',')
        dataset = pd.read_excel(upload_file)
        return dataset
    return None

st.sidebar.header("Proses Data dan Train Model")
berita = upload_dataset()

#nilai C
pilihan_c = [0.1, 0.5, 1, 5, 10]
c = st.sidebar.selectbox("Pilih Nilai C", pilihan_c)

# Menambahkan input untuk memilih algoritma
metode_input = st.sidebar.selectbox("Pilih Metode", [" ", "SVM", "SVM + CS"])

# Menambahkan tombol "Klasifikasi"
klasifikasi_button = st.sidebar.button("Klasifikasi")

# Proses klasifikasi hanya dimulai jika tombol "Klasifikasi" ditekan dan input c dan algoritma sudah diisi
if klasifikasi_button and c and metode_input != "":
     berita,x,y = preprocess_data(berita)
     X_train, X_test, y_train, y_test = split_and_tfidf(berita,x,y)
     svm_cs_results(metode_input,c, X_train,  y_train , X_test, y_test,berita)
    #def svm_cs_results(metode_input, c, X_train, y_train, X_test, y_test, data,model):

st.sidebar.header("Proses Data Test")

def upload_test_dataset():
    upload_file2 = st.file_uploader("Upload Data Test dalam bentuk xlsx", type=["xlsx"], key="upload_test_dataset")
    if upload_file2 is not None:
        dataset = pd.read_excel(upload_file2)
        return dataset
    return None

berita = upload_test_dataset()

# Muat model yang sudah disimpan sebelumnya
selected_c2 = st.sidebar.selectbox("Pilih Nilai C", [0.1, 0.5, 1, 5, 10], key="selectbox_c")
#selected_c2 = st.sidebar.selectbox("Pilih Nilai C", [0.1, 0.5, 1, 5, 10])

# Menambahkan input untuk memilih algoritma
metode_input2 = st.sidebar.selectbox("Pilih Metode", ["SVM", "SVM + CS"])

# Menambahkan tombol "Klasifikasi"
klasifikasi_button_test = st.sidebar.button("Klasifikasi Test SVM")

# Proses klasifikasi hanya dimulai jika tombol "Klasifikasi" ditekan dan input c dan algoritma sudah diisi
if klasifikasi_button_test and selected_c2 :
               
        from joblib import load
        import pandas as pd

        # Memilih model berdasarkan nilai C yang dipilih
        if selected_c2 == 0.1 and metode_input2 == 'SVM':
            model_file = 'svm_model_C01.joblib'
        elif selected_c2 == 0.5 and metode_input2 == 'SVM':
            model_file = 'svm_model_C05.joblib'
        elif selected_c2 == 1 and metode_input2 == 'SVM':
            model_file = 'svm_model_C1.joblib'
        elif selected_c2 == 5 and metode_input2 == 'SVM':
            model_file = 'svm_model_C5.joblib'
        elif selected_c2 == 10 and metode_input2 == 'SVM':
            model_file = 'svm_model_C10.joblib'
        elif selected_c2 == 0.1 and metode_input2 == 'SVM + CS':
            model_file = 'svm_cs_model_C01.joblib'
        elif selected_c2 == 0.5 and metode_input2 == 'SVM + CS':
            model_file = 'svm_cs_model_C05.joblib'
        elif selected_c2 == 1 and metode_input2 == 'SVM + CS':
            model_file = 'svm_cs_model_C1.joblib'
        elif selected_c2 == 5 and metode_input2 == 'SVM + CS':
            model_file = 'svm_cs_model_C5.joblib'
        elif selected_c2 == 10 and metode_input2 == 'SVM + CS':
            model_file = 'svm_cs_model_C10.joblib'
        else:
            #model_file = 'svm_model_C10BARU.joblib'
            print('error')
        # Muat model yang sudah disimpan sebelumnya
        model= load(model_file)
        #model = load('svm_model_C10BARU.joblib')
    
        #df_test = pd.read_excel('Data\\test_data2.xlsx')
        df_test = berita

        df_test = df_test.drop(columns=['Kategori'])

        import pickle

        with open('vectorizer_baru.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        x = df_test['Text Combined']
        df = vectorizer.transform(x)

        prediksi_model = model.predict(df)

        ##MULAI DARI SINI
        st.subheader("Classified Labels and Headlines")

        # Menambahkan kolom 'predicted_label' ke DataFrame
        df_test['predicted_label'] = prediksi_model

        data_kategori_test = pd.read_excel('Data\\y_test.xlsx')

        df_test['real label'] = data_kategori_test['y_test']

        # Menambahkan kolom 'is_correct' yang berisi label true jika 'predicted_label' sama dengan 'true_label'
        df_test['is_correct'] = df_test['predicted_label'] == data_kategori_test['y_test']

        st.subheader("Data Test")
        st.write(df_test)

# Mengatur warna teks output
st.markdown('<style>h1{color: #008B8B;}</style>', unsafe_allow_html=True)

