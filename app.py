
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Prediksi Risiko Stroke", layout="centered", page_icon="🏥")

st.title("🏥 Aplikasi Prediksi Risiko Penyakit Stroke")
st.write("Aplikasi Interaktif Tugas Besar - Kelompok 8")
st.markdown("---")

@st.cache_resource
def load_resources():
    with open('model_rf.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_resources()
except FileNotFoundError:
    st.error("File 'model_rf.pkl' tidak ditemukan! Pastikan cell pengecekan di notebook sudah sukses dieksekusi.")

st.header("Form Input Data Medis Pasien")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    age = st.number_input("Usia (Tahun)", min_value=0, max_value=120, value=45)
    hypertension = st.selectbox("Memiliki Riwayat Hipertensi?", ["Tidak", "Ya"])
    heart_disease = st.selectbox("Memiliki Penyakit Jantung?", ["Tidak", "Ya"])
    ever_married = st.selectbox("Pernah Menikah?", ["No", "Yes"])
    residence = st.selectbox("Tipe Tempat Tinggal", ["Urban", "Rural"])

with col2:
    avg_glucose = st.number_input("Rata-Rata Kadar Glukosa (mg/dL)", min_value=0.0, value=100.0)
    bmi = st.number_input("Indeks Massa Tubuh / BMI", min_value=0.0, value=25.0)
    work_type = st.selectbox("Tipe Pekerjaan", ["Govt_job", "Never_worked", "Private", "Self-employed", "children"])
    smoking_status = st.selectbox("Status Merokok", ["Unknown", "formerly smoked", "never smoked", "smokes"])

if st.button("Analisis Risiko Stroke", type="primary"):
    # Encoding biner (0 atau 1)
    gender_encoded = 1 if gender == "Male" else 0
    married_encoded = 1 if ever_married == "Yes" else 0
    residence_encoded = 1 if residence == "Urban" else 0
    hypertension_encoded = 1 if hypertension == "Ya" else 0
    heart_disease_encoded = 1 if heart_disease == "Ya" else 0
    
    # Struktur data input yang disamakan persis dengan format DataFrame pasca get_dummies
    data_input = {
        'gender': gender_encoded,
        'age': age,
        'hypertension': hypertension_encoded,
        'heart_disease': heart_disease_encoded,
        'ever_married': married_encoded,
        'Residence_type': residence_encoded,
        'avg_glucose_level': avg_glucose,
        'bmi': bmi,
        
        # One-hot encoding untuk work_type (diubah ke integer 1/0 agar tidak terbaca Boolean)
        'work_type_Govt_job': int(work_type == "Govt_job"),
        'work_type_Never_worked': int(work_type == "Never_worked"),
        'work_type_Private': int(work_type == "Private"),
        'work_type_Self-employed': int(work_type == "Self-employed"),
        'work_type_children': int(work_type == "children"),
        
        # One-hot encoding untuk smoking_status
        'smoking_status_Unknown': int(smoking_status == "Unknown"),
        'smoking_status_formerly smoked': int(smoking_status == "formerly smoked"),
        'smoking_status_never smoked': int(smoking_status == "never smoked"),
        'smoking_status_smokes': int(smoking_status == "smokes")
    }
    
    df_input = pd.DataFrame([data_input])
    
    # Menyamakan tipe data seluruh kolom menjadi float/int agar pas dengan input Random Forest
    df_input = df_input.astype(float)
    
    # Jalankan prediksi
    prediction = model.predict(df_input)
    proba = model.predict_proba(df_input)[0][1] * 100
    
    st.markdown("---")
    st.subheader("Hasil Analisis Klinis:")
    if prediction[0] == 1:
        st.error(f"⚠️ **RISIKO TINGGI:** Pasien terindikasi memiliki risiko penyakit stroke.")
    else:
        st.success(f"✅ **RISIKO RENDAH:** Pasien terindikasi aman dari risiko penyakit stroke.")
    st.write(f"Probabilitas kecenderungan model: **{proba:.2f}%**")
