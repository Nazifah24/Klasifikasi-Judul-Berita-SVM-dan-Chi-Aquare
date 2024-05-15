import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score, 
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from Klasifikasi import casefolding,cleaning,tokenizing,tfidf,klasifikasiSVM,feature_selection_chi2

def preprocess_data(data):      #Ini adalah deklarasi fungsi preprocess_data yang menerima satu parameter yaitu data
    #data = data.drop(columns=['Article Link'])
    data = data.drop(columns=['Unnamed: 0','Article Link'])

    # berita

    # @title Default title text
    data = data[(data['Kategori'] != 'NEWS') & (data['Kategori'] != 'HOT')]

    # Inisialisasi LabelEncoder
    label_encoder = LabelEncoder()

    # Mengubah kolom 'kategori' menjadi angka
    data['kategori_encoded'] = label_encoder.fit_transform(data['Kategori'])

    # Menghapus baris dengan kategori 'NEWS' atau 'HOT'
    data = data[(data['Kategori'] != 'NEWS') & (data['Kategori'] != 'HOT')]

    # Menggabungkan Kolom Article Title dan Article Content
    data['Text Combined'] = data['Article Title'] + " " + data['Article Content']

    # Membersihkan kolom 'Article Title' dari karakter yang tidak diinginkan
    # data['Article Title'] = data['Article Title'].apply(lambda x: x.strip().replace('\n', ''))
    # data['Article Content'] = data['Article Content'].apply(lambda x: x.strip().replace('\n', ''))
    data['Text Combined'] = data['Text Combined'].apply(lambda x: x.strip().replace('\n', ''))
    # Menghilangkan spasi berlebih
    data['Text Combined'] = data['Text Combined'].apply(lambda x: ' '.join(x.split()))

    from langdetect import detect

    def filter_indonesian_text(text):
        try:
            if detect(text) == 'id':
                return text
            else:
                return ''
        except:
            return ''

    # Mengaplikasikan fungsi filter ke kolom 'Text Combined'
    data['Text Combined'] = data['Text Combined'].apply(filter_indonesian_text)

    # Menghapus baris yang kosong (teks bukan dalam bahasa Indonesia)
    data = data[data['Text Combined'] != '']

    # data["Article Title"] = data["Article Title"]

    st.subheader("Data setelah di preprocess")  #Ini adalah penggunaan fungsi subheader dari st untuk menampilkan subjudul di aplikasi web. Dalam hal ini, subjudulnya adalah "Data setelah di preprocess".
    st.subheader("Cleaning ")   #Menampilkan subjudul "Cleaning" menggunakan fungsi subheader dari st.
    # data["Article Title"] = data["Article Title"].apply(cleaning)
    # data["Article Content"] = data["Article Content"].apply(cleaning)
    data["Text Combined"] = data["Text Combined"].apply(cleaning)

    st.write(data)      #Menampilkan DataFrame data setelah dilakukan preprocessing dan pembersihan pada kolom "Article Title". Fungsi write dari st digunakan untuk menampilkan informasi di aplikasi web.

    st.subheader("Casefolding ")
    # data["Article Title"] = data["Article Title"].apply(casefolding)
    # data["Article Content"] = data["Article Content"].apply(casefolding)
    data["Text Combined"] = data["Text Combined"].apply(casefolding)

    st.write(data)

    st.subheader("Tokenizing")
    # data["Article Title"] = data["Article Title"].apply(tokenizing)
    # data["Article Content"] = data["Article Content"].apply(tokenizing)
    data["Text Combined"] = data["Text Combined"].apply(tokenizing)

    st.write(data)

    all_words = []
    for tokenized_title in data["Text Combined"]:
        all_words.extend(tokenized_title)

    # Mencari unique words
    unique_words = set(all_words)

    # Menghitung jumlah unique words
    jumlah_unique_words = len(unique_words)

    # Menampilkan unique words dan jumlahnya
    #print("Unique Words di Article Title:", unique_words)
    print("Jumlah Unique Words di Text Combined :", jumlah_unique_words)

    #x = data[['Article Title', 'Article Content']]
    x = data['Text Combined']
    y = data['kategori_encoded']

    print(len(x))
    print(len(y))
    
    return data,x,y

def split_and_tfidf(data,x,y):
    data_tfidf = tfidf(data)

    X_train, X_test, y_train, y_test = train_test_split(
        #data_tfidf, data["kategori_encoded"], test_size=0.2, random_state=42
        data_tfidf,y,test_size=0.2, random_state=42
    )

    print("Jumlah Text Data Latih = " , X_train.shape[0])
    print("Jumlah Label Data Latih = " , y_train.shape[0])
    print("Jumlah Text Data Testing = " , X_test.shape[0])
    print("Jumlah Label Data Testing = " , y_test.shape[0])

    # Hitung jumlah sampel untuk setiap kelas di data test
    class_counts = y_test.value_counts()

    # Tampilkan hasil
    print("Jumlah sampel untuk setiap kelas di data test:")
    print(class_counts)

    import streamlit as st
    import matplotlib.pyplot as plt

    # Hitung nilai-nilai dan frekuensinya dari y_train
    
    def plot(jenis_data , counts):
        # Buat bar chart dengan warna yang berbeda untuk setiap bar
        plt.figure(figsize=(8, 6))
        counts.plot(kind='bar', color=['blue', 'green', 'red'])  # Atur warna sesuai dengan jumlah kategori yang Anda miliki
        plt.xlabel('Kategori')
        plt.ylabel('Frekuensi')
        plt.title('Bar Chart dari Data ' + jenis_data)
        plt.xticks(rotation=0)
        # Tampilkan plot menggunakan Streamlit
        st.pyplot(plt)

    plot(jenis_data ='Train', counts = y_train.value_counts())
    plot(jenis_data ='Test' ,counts = y_test.value_counts())

    return X_train, X_test, y_train, y_test

def svm_cs_results(metode_input, c, X_train, y_train, X_test, y_test, data):
    if metode_input == "SVM":
        predicted = klasifikasiSVM(metode_input,c,X_train, y_train, X_test, y_test)

        st.text("SVM Metrics Data Training:")
        st.text(f"SVM Accuracy: {accuracy_score(y_train, klasifikasiSVM(metode_input,c, X_train, y_train, X_train, y_train))}")
        st.text(f"SVM Precision: {precision_score(y_train, klasifikasiSVM(metode_input,c, X_train, y_train, X_train, y_train), average='weighted')}")
        st.text(f"SVM Recall: {recall_score(y_train, klasifikasiSVM(metode_input,c, X_train, y_train, X_train, y_train), average='weighted')}")
        st.text(f"SVM F1 Score: {f1_score(y_train, klasifikasiSVM(metode_input,c, X_train, y_train, X_train, y_train), average='weighted')}")
        st.text("Confusion Matrix Data Training:")
        st.write(confusion_matrix(y_train, klasifikasiSVM(metode_input,c, X_train, y_train, X_train, y_train)))

        report_train = classification_report(y_train, klasifikasiSVM(metode_input,c, X_train, y_train, X_train, y_train))
        st.write("Classification Report Data Training")
        st.code(report_train, language='markdown')

        st.subheader("Classified Labels and Headlines Data Testing")
        st.write("Jika 2 = SPORT, 1 = FINANCE dan 0 = EDU")
    
        st.text("SVM Metrics Data Testing:")
        st.text(f"SVM Accuracy: {accuracy_score(y_test, predicted)}")
        st.text(f"SVM Precision: {precision_score(y_test, predicted, average='weighted')}")
        st.text(f"SVM Recall: {recall_score(y_test, predicted, average='weighted')}")
        st.text(f"SVM F1 Score: {f1_score(y_test, predicted, average='weighted')}")
        st.text("Confusion Matrix Data Testing:")
        st.write(confusion_matrix(y_test, predicted))

        report_test = classification_report(y_test, predicted)
        st.write("Classification Report Data Testing")
        st.code(report_test, language='markdown')

        from joblib import load
        import pandas as pd

         # Memilih model berdasarkan nilai C yang dipilih
        if c == 0.1 and metode_input == 'SVM':
            model_file = 'svm_model_C01.joblib'
        elif c == 0.5 and metode_input == 'SVM':
            model_file = 'svm_model_C05.joblib'
        elif c == 1 and metode_input == 'SVM':
            model_file = 'svm_model_C1.joblib' 
        elif c == 5 and metode_input == 'SVM':
            model_file = 'svm_model_C5.joblib'
        elif c == 10 and metode_input == 'SVM':
            model_file = 'svm_model_C10.joblib'
        else:
            #model_file = 'svm_model_C10BARU.joblib'
            print('error')

        model = load(model_file)

        # Muat model yang sudah disimpan sebelumnya
        #model = load('svm_model_C05BARU.joblib')
        import pandas as pd
        df_test = pd.read_excel('Data\\test_data3.xlsx')

        df_test = df_test.drop(columns=['Kategori'])

        import pickle

        with open('vectorizer_baru.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        x = df_test['Text Combined']
        df = vectorizer.transform(x)

        prediksi_model = model.predict(df)

        ##MULAI DARI SINI
        st.subheader("Classified Labels and Headlines")
        import pandas as pd

        # Menambahkan kolom 'predicted_label' ke DataFrame
        df_test['predicted_label'] = prediksi_model

        data_kategori_test = pd.read_csv('Data\\y_test.csv')

        df_test['real label'] = data_kategori_test['y_test']

        # Menambahkan kolom 'is_correct' yang berisi label true jika 'predicted_label' sama dengan 'true_label'
        df_test['is_correct'] = df_test['predicted_label'] == data_kategori_test['y_test']

        st.subheader("Data Test")
        st.write(df_test)


    # SVM Dengan Chi Square
    elif metode_input == "SVM + CS":
        import pandas as pd
        st.subheader("SVM With Chi Square")
        
        X_train_res, X_test_res = feature_selection_chi2(X_train, y_train,X_test)
        predicted = klasifikasiSVM(metode_input,c,X_train_res, y_train, X_test_res, y_test)

        st.text("SVM Metrics Data Training:")
        st.text(f"SVM Accuracy: {accuracy_score(y_train, klasifikasiSVM(metode_input,c, X_train, y_train, X_train, y_train))}")
        st.text(f"SVM Precision: {precision_score(y_train, klasifikasiSVM(metode_input,c, X_train, y_train, X_train, y_train), average='weighted')}")
        st.text(f"SVM Recall: {recall_score(y_train, klasifikasiSVM(metode_input,c, X_train, y_train, X_train, y_train), average='weighted')}")
        st.text(f"SVM F1 Score: {f1_score(y_train, klasifikasiSVM(metode_input,c, X_train, y_train, X_train, y_train), average='weighted')}")
        st.text("Confusion Matrix Data Training:")
        st.write(confusion_matrix(y_train, klasifikasiSVM(metode_input,c, X_train, y_train, X_train, y_train)))

        report_train = classification_report(y_train, klasifikasiSVM(metode_input,c, X_train, y_train, X_train, y_train))
        st.write("Classification Report Data Training")
        st.code(report_train, language='markdown')

        st.subheader("Classified Labels and Headlines Data Testing")

        st.text("SVM Metrics Data Testing:")
        st.text(f"SVM Accuracy: {accuracy_score(y_test, predicted)}")
        st.text(f"SVM Precision: {precision_score(y_test, predicted, average='weighted')}")
        st.text(f"SVM Recall: {recall_score(y_test, predicted, average='weighted')}")
        st.text(f"SVM F1 Score: {f1_score(y_test, predicted, average='weighted')}")
        st.text("Confusion Matrix Data Testing:")
        st.write(confusion_matrix(y_test, predicted))

        report_test = classification_report(y_test, predicted)
        st.write("Classification Report Data Testing")
        st.code(report_test, language='markdown')

        from joblib import load
        import pandas as pd

        # Muat model yang sudah disimpan sebelumnya
        if c == 0.1 and metode_input == 'SVM + CS':
            model_file = 'svm_cs_model_C01.joblib'
        elif c == 0.5 and metode_input == 'SVM + CS':
            model_file = 'svm_cs_model_C05.joblib'
        elif c == 1 and metode_input == 'SVM + CS':
            model_file = 'svm_cs_model_C1.joblib' 
        elif c == 5 and metode_input == 'SVM + CS':
            model_file = 'svm_cs_model_C5.joblib'
        elif c == 10 and metode_input == 'SVM + CS':
            model_file = 'svm_cs_model_C10.joblib'

        model = load(model_file)

        import pandas as pd
        df_test = pd.read_excel('Data\\test_data3.xlsx')

        df_test = df_test.drop(columns=['Kategori'])

        import pickle

        with open('vectorizer_baru.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        x = df_test['Text Combined']
        df = vectorizer.transform(x)

        prediksi_model = model.predict(df)

        ##MULAI DARI SINI
        st.subheader("Classified Labels and Headlines")
        import pandas as pd

        # Menambahkan kolom 'predicted_label' ke DataFrame
        df_test['predicted_label'] = prediksi_model

        data_kategori_test = pd.read_csv('Data\\y_test.csv')

        df_test['real label'] = data_kategori_test['y_test']

        # Menambahkan kolom 'is_correct' yang berisi label true jika 'predicted_label' sama dengan 'true_label'
        df_test['is_correct'] = df_test['predicted_label'] == data_kategori_test['y_test']

        st.subheader("Data Test")
        st.write(df_test)
