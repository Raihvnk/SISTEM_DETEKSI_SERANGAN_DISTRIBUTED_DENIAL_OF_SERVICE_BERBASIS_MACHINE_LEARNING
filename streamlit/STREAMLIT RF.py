import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np

# Membaca dataset
@st.cache_data
def load_data():
    train_dataset = pd.read_csv('preprocessed_train_datasets.csv')
    test_dataset = pd.read_csv('preprocessed_test_datasets.csv')
    return train_dataset, test_dataset

train_dataset, test_dataset = load_data()

# Memisahkan fitur dan label
X_train = train_dataset.drop('label', axis=1)
y_train = train_dataset['label']
X_test = test_dataset.drop('label', axis=1)
y_test = test_dataset['label']

# Streamlit App Title
st.title("DDoS Attack Detection Using Random Forest")

# Fungsi untuk mengambil input dari pengguna
def get_user_input():
    datapath_id = st.number_input("Masukkan datapath_id:", min_value=0, value=0)
    version = st.number_input("Masukkan version:", min_value=0, value=0)
    header_length = st.number_input("Masukkan header_length:", min_value=0, value=0)
    total_length = st.number_input("Masukkan total_length:", min_value=0, value=0)
    ttl = st.number_input("Masukkan ttl:", min_value=0, value=0)
    proto = st.number_input("Masukkan proto:", min_value=0, value=0)
    csum = st.number_input("Masukkan csum:", min_value=0, value=0)
    src_port = st.number_input("Masukkan src_port:", min_value=0, value=0)
    dst_port = st.number_input("Masukkan dst_port:", min_value=0, value=0)
    tcp_flag = st.number_input("Masukkan tcp_flag:", min_value=0, value=0)
    type_icmp = st.number_input("Masukkan type_icmp:", min_value=0, value=0)
    csum_icmp = st.number_input("Masukkan csum_icmp:", min_value=0, value=0)
    port_no = st.number_input("Masukkan port_no:", min_value=0, value=0)
    rx_bytes_ave = st.number_input("Masukkan rx_bytes_ave:", min_value=0, value=0)
    tx_bytes_ave = st.number_input("Masukkan tx_bytes_ave:", min_value=0, value=0)

    data_input = {
        'datapath_id': datapath_id,
        'version': version,
        'header_length': header_length,
        'total_length': total_length,
        'ttl': ttl,
        'proto': proto,
        'csum': csum,
        'src_port': src_port,
        'dst_port': dst_port,
        'tcp_flag': tcp_flag,
        'type_icmp': type_icmp,
        'csum_icmp': csum_icmp,
        'port_no': port_no,
        'rx_bytes_ave': rx_bytes_ave,
        'tx_bytes_ave': tx_bytes_ave,
    }

    features_input = pd.DataFrame(data_input, index=[0])
    return features_input

# Mendapatkan input dari pengguna
user_input = get_user_input()

# Tampilkan input pengguna
st.write("User Input Features:")
st.write(user_input)

# Melatih model Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Prediksi dan probabilitas berdasarkan input pengguna
user_pred_proba = rf_model.predict_proba(user_input)

# Mapping prediksi ke label kelas
label_mapping = {0: "DDOS_ICMP", 1: "DDOS_TCP", 2: "DDOS_UDP", 3: "NORMAL_ICMP", 4: "NORMAL_TCP", 5: "NORMAL_UDP"}

# Menentukan prediksi dengan probabilitas tertinggi
user_pred = np.argmax(user_pred_proba, axis=1)[0]
prediction_label = label_mapping[user_pred]

# Tampilkan hasil prediksi
st.write(f"Prediksi Kelas: {prediction_label}")
st.write("Probabilitas Kelas:")
st.write(user_pred_proba)

# Streamlit Input for Number of Splits in K-Fold
n_splits = st.sidebar.slider('Number of K-Fold Splits', min_value=2, max_value=10, value=5)

# Metode K-Fold Cross-Validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Inisialisasi variabel untuk menyimpan hasil evaluasi
accuracy_scores = []

# Loop melalui setiap fold
for train_index, val_index in kf.split(X_train):
    # Membagi data ke dalam data training dan data validasi
    X_train_kfold, X_val_kfold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_kfold, y_val_kfold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # Membuat dan melatih model Random Forest
    rf_model_kfold = RandomForestClassifier()
    rf_model_kfold.fit(X_train_kfold, y_train_kfold)
    
    # Evaluasi model pada data validasi
    accuracy_scores.append(rf_model_kfold.score(X_val_kfold, y_val_kfold))

# Menghitung rata-rata hasil evaluasi
mean_accuracy = np.mean(accuracy_scores)

# Tampilkan hasil evaluasi
st.write("K-Fold Cross-Validation Mean Accuracy: ", mean_accuracy)

# Evaluasi akhir menggunakan test_dataset
st.write("\nFinal Evaluation on Test Dataset:")

# Membuat prediksi menggunakan test_dataset
final_accuracy = rf_model.score(X_test, y_test)

# Tampilkan hasil evaluasi pada test data
st.write("Test Dataset Accuracy:")
st.write(final_accuracy)
