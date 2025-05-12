# 🚀 Sistem Deteksi Serangan DDoS Berbasis Machine Learning

Proyek ini merupakan bagian dari skripsi yang mengimplementasikan algoritma machine learning untuk mendeteksi serangan Distributed Denial of Service (DDoS) dalam lingkungan Software Defined Networking (SDN).

---

## 📚 Dataset

Dataset yang digunakan berasal dari [Mendeley Data – SDN-DDoS (ICMP, TCP, UDP)](https://data.mendeley.com/datasets/hkjbp67rsc/1), dipublikasikan oleh:

- **Oxicusa Gugi Housman**, **Hafida Isnaini**, **Fauzi Dwi Setiawan Sumadi**  
- *Universitas Muhammadiyah Malang (2020)*  
- DOI: [10.17632/hkjbp67rsc.1](https://doi.org/10.17632/hkjbp67rsc.1)  
- Lisensi: Creative Commons **CC BY 4.0**

### 🔍 Deskripsi Dataset:

- Berisi lalu lintas jaringan normal dan serangan DDoS (ICMP, TCP, UDP flood)
- Topologi: `tree, depth=3, fanout=2` dengan controller RYU
- Serangan disimulasikan dari H1–H4 menuju H4 menggunakan **Scapy** dan **TCPReplay**
- Data terdiri dari `.pcap` hasil packet generation dan `.csv` hasil ekstraksi fitur dari header dan statistik port melalui RYU Controller yang telah dimodifikasi

---

## 🤖 Model Machine Learning

### Algoritma yang Diuji:
- SVM (Support Vector Machine)
- Naive Bayes
- Decision Tree
- CNN (Convolutional Neural Network)
- Random Forest

Model-model tersebut diuji dari sisi akurasi dan waktu eksekusi.

### ✅ Model Implementasi Akhir:
- **Random Forest**
- **K-Fold Cross Validation**

> Dipilih karena akurasi paling tinggi, stabil, dan waktu eksekusi cepat — cocok digunakan di komputer lokal tempat penelitian dilakukan.

---

## 🧪 Pengujian Sistem

### 1. **GUI Lokal dengan Streamlit**
- Antarmuka pengguna interaktif untuk menguji input data terhadap model
- 🎥 [Demo Streamlit – YouTube](https://youtu.be/glfzqsn_Zhs)

### 2. **Simulasi Real-Time dengan Mininet + RYU Controller**
- Simulasi SDN untuk uji deteksi serangan langsung
- 🎥 [Demo RYU Controller & Mininet – YouTube](https://youtu.be/8tqxjv_XoXI)

---

## 🛠 Teknologi yang Digunakan

| Komponen         | Fungsi                               |
|------------------|----------------------------------------|
| Python           | Bahasa utama                          |
| Scikit-learn     | Machine Learning (Random Forest, dst) |
| Streamlit        | GUI untuk pengujian lokal              |
| Ryu Controller   | SDN Controller                         |
| Mininet          | Emulator jaringan                      |
| Wireshark        | Monitoring lalu lintas jaringan        |
| Scapy & TCPReplay| Generator traffic serangan             |

---
