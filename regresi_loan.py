import streamlit as st
import pandas as pd
import joblib # Menggunakan joblib

# Memuat model regresi yang sudah dilatih
try:
    with open('regresi.joblib', 'rb') as file: # Sesuaikan nama file jika Anda menyimpannya sebagai .pkl
        model = joblib.load(file)
except FileNotFoundError:
    st.error("File 'regresi.joblib' tidak ditemukan. Pastikan model berada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# Judul aplikasi Streamlit
st.title('Aplikasi Prediksi Income')
st.write('Aplikasi ini memprediksi Income berdasarkan Age (Usia) dan Experience (Pengalaman).')

# Input dari pengguna
st.sidebar.header('Input Data Baru')
age = st.sidebar.slider('Age (Usia)', min_value=18, max_value=80, value=30)
experience = st.sidebar.slider('Experience (Pengalaman)', min_value=0, max_value=60, value=5)

# Menampilkan input yang diterima
st.write(f"**Data yang dimasukkan:**")
st.write(f"Usia: {age} tahun")
st.write(f"Pengalaman: {experience} tahun")

# Tombol untuk melakukan prediksi
if st.button('Prediksi Income'):
    try:
        # Buat DataFrame dari input baru dengan nama kolom yang sama seperti saat training
        new_data_df = pd.DataFrame([[age, experience]], columns=['Age', 'Experience'])

        # Lakukan prediksi menggunakan model yang sudah dilatih
        predicted_income = model.predict(new_data_df)

        st.success(f"**Prediksi Income Anda adalah: ${predicted_income[0][0]:,.2f}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

st.markdown("""
---
*Catatan: Pastikan model `regresi.joblib` sudah diserialisasi dengan benar dan mengandung model regresi yang siap digunakan.*
""")