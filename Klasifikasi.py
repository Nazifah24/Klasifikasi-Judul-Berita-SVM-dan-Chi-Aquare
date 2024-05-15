import nltk
import re
from sklearn import svm
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import pickle

nltk.download('punkt')

def cleaning(data):
     # Menghapus karakter khusus
    data = re.sub(r'[^\w\s]', '', data)

    #Menghapus angka
    data = re.sub(r'\d+', '', data)

    return data

def casefolding(data):
     # Mengubah huruf menjadi huruf kecil
    data = data.lower()
    return data

def tokenizing(data):
    return word_tokenize(data)

def tfidf(data):    #Ini adalah deklarasi fungsi tfidf yang menerima satu parameter yaitu data
    tfidf_vectorizer = TfidfVectorizer()     #Membuat objek TfidfVectorizer yang akan digunakan untuk menghitung TF-IDF.
    text_tf = tfidf_vectorizer.fit_transform(data['Text Combined'].astype("U"))   #Menggunakan TfidfVectorizer untuk menghitung TF-IDF dari teks dalam kolom "Article Title" dari DataFrame data.
                                                                    #.astype("U") digunakan untuk memastikan bahwa teks dianggap sebagai Unicode.
                                                                    #fit_transform digunakan untuk menghitung dan mengubah data teks menjadi representasi TF-IDF.
                                                                    #Hasilnya disimpan dalam variabel text_tf.
    
    # Simpan objek TfidfVectorizer ke dalam file pickle
    with open('vectorizer_baru.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    return text_tf


def klasifikasiSVM(metode_input,c,X_train, y_train, X_hold, y_test):   #Ini adalah deklarasi fungsi klasifikasiSVM yang menerima lima parameter: c (nilai hyperparameter C untuk SVM), X_train (data latih), y_train (label data latih), X_hold (data uji), dan y_test (label data uji).
    olf = svm.SVC(kernel="linear", gamma="scale", C=c).fit(X_train, y_train)   #Membuat objek Support Vector Classification (SVC) dengan kernel linear menggunakan svm.SVC.
                                                                               #Parameter kernel="linear" menunjukkan bahwa model SVM menggunakan kernel linear.
                                                                               #Parameter gamma="scale" menunjukkan bahwa nilai gamma dihitung secara otomatis berdasarkan skala data.
                                                                               #Parameter C=c menunjukkan nilai hyperparameter C yang diberikan oleh parameter fungsi.
                                                                               #Menggunakan data latih (X_train dan y_train) untuk melatih model dengan memanggil metode .fit().
    predicted = olf.predict(X_hold)   #Menggunakan model yang telah dilatih (olf) untuk melakukan prediksi terhadap data uji (X_hold) dengan memanggil metode .predict(). hasil disimpan dalam variabel 'predict'
    
    from joblib import dump

     # Simpan model ke file joblib
    if c == 0.1 and metode_input == 'SVM':
        dump(olf, 'svm_model_C01.joblib')
    elif c == 0.5 and metode_input == 'SVM':
        dump(olf, 'svm_model_C05.joblib')
    elif c == 1 and metode_input == 'SVM':
        dump(olf, 'svm_model_C1.joblib')
    elif c == 5 and metode_input == 'SVM':
        dump(olf, 'svm_model_C5.joblib')
    elif c == 10 and metode_input == 'SVM':
        dump(olf, 'svm_model_C10.joblib')
    elif c == 0.1 and metode_input == 'SVM + CS':
        dump(olf, 'svm_cs_model_C01.joblib')
    elif c == 0.5 and metode_input == 'SVM + CS':
        dump(olf, 'svm_cs_model_C05.joblib')
    elif c == 1 and metode_input == 'SVM + CS':
        dump(olf, 'svm_cs_model_C1.joblib')
    elif c == 5 and metode_input == 'SVM + CS':
        dump(olf, 'svm_cs_model_C5.joblib')
    elif c == 10 and metode_input == 'SVM + CS':
        dump(olf, 'svm_cs_model_C10.joblib')
    
    return predicted    #Mengembalikan nilai prediksi dari model SVM untuk data uji.


def feature_selection_chi2(X_train, y_train, X_test, k_best=2000):     #Ini adalah deklarasi fungsi feature_selection_chi2 yang menerima empat parameter: X_train (data latih), y_train (label data latih), X_test (data uji), dan k_best (jumlah fitur terbaik yang akan dipilih, defaultnya adalah 5 jika tidak diberikan).
    selector = SelectKBest(chi2, k=k_best)          #Membuat objek SelectKBest yang menggunakan metode chi2 (Chi-Square) sebagai fungsi skor untuk melakukan seleksi fitur.
                                                    #Parameter k=k_best menentukan jumlah fitur terbaik yang akan dipilih.
    X_train_selected = selector.fit_transform(X_train, y_train)   #Menggunakan objek selector untuk melakukan seleksi fitur pada data latih (X_train) dengan memanggil metode .fit_transform()
                                                                  #Hasilnya adalah X_train_selected, yaitu data latih yang hanya terdiri dari fitur terbaik yang telah dipilih.
    X_test_selected = selector.transform(X_test)                #Menggunakan objek selector yang sama untuk melakukan seleksi fitur pada data uji (X_test) dengan memanggil metode .transform().
                                                                #Hasilnya adalah X_test_selected, yaitu data uji yang hanya terdiri dari fitur terbaik yang telah dipilih.
    return X_train_selected, X_test_selected            #Mengembalikan data latih dan data uji yang sudah melalui proses seleksi fitur menggunakan metode Chi-Square.
