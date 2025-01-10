# Mengimpor library yang diperlukan
import base64  # Untuk mengonversi data biner menjadi format base64
import numpy as np  # Untuk operasi numerik dan manipulasi array
import streamlit as st  # Untuk membuat aplikasi web menggunakan Streamlit
from joblib import load  # Untuk memuat model dan encoder yang telah disimpan

import os  # Untuk manipulasi file dan operasi sistem


# Path ke gambar yang akan dimasukkan ke dalam aplikasi Streamlit
image_path = "./images/background-streamlit.png"

# Membuka gambar dalam mode biner dan mengonversinya ke format base64
with open(image_path, "rb") as image_file:
    # Membaca konten gambar, mengonversinya ke format Base64, dan mendekode hasilnya menjadi string
    encoded_string = base64.b64encode(image_file.read()).decode()

# Gaya CSS dengan gambar Base64 sebagai latar belakang untuk aplikasi Streamlit
background_image = f"""
<style>
/* Menambahkan latar belakang dengan gambar Base64 pada container utama */
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{encoded_string}");  /* Gambar dalam format base64 */
    background-size: cover;  /* Memastikan gambar menutupi seluruh area */
    background-position: center;  /* Menjaga posisi gambar tetap di tengah */
    background-repeat: no-repeat;  /* Mencegah gambar diulang */
    background-attachment: fixed;  /* Mengunci posisi gambar agar tidak bergerak saat menggulir halaman */
}}
/* Menambahkan gaya untuk sidebar agar transparan */
[data-testid="stSidebar"] {{
    background: rgba(0,0,0,0);  /* Membuat sidebar transparan */
}}
/* Menambahkan gaya untuk tombol-tombol */
button[role="button"] {{
    color: white;  /* Mengatur warna teks tombol */
    background-color: black;  /* Mengatur warna latar belakang tombol */
    border: 2px solid black;  /* Menambahkan border pada tombol */
    padding: 10px 20px;  /* Mengatur padding tombol */
    border-radius: 5px;  /* Mengatur kelengkungan sudut tombol */
    font-size: 16px;  /* Mengatur ukuran font tombol */
}}
button[role="button"]:hover {{
    color: black;  /* Mengubah warna teks saat hover */
    background-color: white;  /* Mengubah warna latar belakang tombol saat hover */
    border-color: black;  /* Mengubah warna border saat hover */
}}
button[role="button"]:active {{
    color: black;  /* Mengubah warna teks saat tombol ditekan */
    background-color: white;  /* Mengubah warna latar belakang saat tombol ditekan */
    border-color: black;  /* Mengubah warna border saat tombol ditekan */
}}
/* Mengatur warna header */
h1, h2, h3, h4, h5, h6 {{
    color: black;  /* Menetapkan warna header ke hitam */
}}
/* Mengatur warna label untuk input */
.stTextInput label, .stSelectbox label, .stNumberInput label {{
    color: black;  /* Warna untuk label input seperti number input dan selectbox */
}}
/* Mengatur warna teks output */
.stMarkdown, .stWrite, .stText {{
    color: black;  /* Mengubah warna teks output menjadi hitam */
}}
</style>
"""

# Menerapkan gaya CSS yang telah dibuat dengan background gambar ke aplikasi Streamlit
st.markdown(background_image, unsafe_allow_html=True)

# Memuat model dan label encoder yang telah disimpan
try:
    # Memuat model RandomForestClassifier yang telah disimpan
    model = load('diagnosis_predictor.joblib')
    # Memuat label encoder untuk menerjemahkan label prediksi kembali ke nama yang sesungguhnya
    label_encoder = load('label_encoder.joblib')
    
    # Ambil akurasi dan nama model dari metadata di objek model
    model_accuracy = model.score if hasattr(model, "score") else 0.93  # Jika tidak ada metode score, gunakan nilai default
    model_name = type(model).__name__  # Mendapatkan nama jenis model, seperti "RandomForestClassifier"
except Exception as e:
    # Menampilkan pesan error jika model atau label encoder gagal dimuat
    st.error(f"Error loading model or label encoder: {e}")
    st.stop()  # Menghentikan aplikasi jika ada kesalahan saat memuat model


# Mendefinisikan daftar lengkap gejala yang dapat digunakan untuk diagnosa penyakit
symptoms = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills", "joint_pain",
    "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition",
    "spotting_urination", "fatigue", "weight_gain", "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss",
    "restlessness", "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", "high_fever", "sunken_eyes",
    "breathlessness", "sweating", "dehydration", "indigestion", "headache", "yellowish_skin", "dark_urine", "nausea",
    "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation", "abdominal_pain", "diarrhoea",
    "mild_fever", "yellow_urine", "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach",
    "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm", "throat_irritation", "redness_of_eyes",
    "sinus_pressure", "runny_nose", "congestion", "chest_pain", "weakness_in_limbs", "fast_heart_rate",
    "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness",
    "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes", "enlarged_thyroid",
    "brittle_nails", "swollen_extremeties", "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips",
    "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints",
    "movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness", "weakness_of_one_body_side",
    "loss_of_smell", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine", "passage_of_gases",
    "internal_itching", "toxic_look_(typhos)", "depression", "irritability", "muscle_pain", "altered_sensorium",
    "red_spots_over_body", "belly_pain", "abnormal_menstruation", "dischromic _patches", "watering_from_eyes",
    "increased_appetite", "polyuria", "family_history", "mucoid_sputum", "rusty_sputum", "lack_of_concentration",
    "visual_disturbances", "receiving_blood_transfusion", "receiving_unsterile_injections", "coma", "stomach_bleeding",
    "distention_of_abdomen", "history_of_alcohol_consumption", "fluid_overload.1", "blood_in_sputum",
    "prominent_veins_on_calf", "palpitations", "painful_walking", "pus_filled_pimples", "blackheads", "scurring",
    "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails", "blister",
    "red_sore_around_nose", "yellow_crust_ooze"
]

# Dictionary untuk rekomendasi obat berdasarkan diagnosis
medication_db = {
    "Fungal infection": ["Clotrimazole", "Fluconazole"],
    "Allergy": ["Loratadine", "Cetirizine"],
    "GERD": ["Omeprazole", "Esomeprazole"],
    "Chronic cholestasis": ["Ursodiol", "Cholestyramine"],
    "Drug Reaction": ["Antihistamines", "Corticosteroids"],
    "Peptic ulcer disease": ["Ranitidine", "Pantoprazole"],
    "AIDS": ["Antiretroviral Therapy (ART)", "Zidovudine"],
    "Diabetes": ["Metformin", "Insulin"],
    "Gastroenteritis": ["Oral Rehydration Salts", "Loperamide"],
    "Bronchial Asthma": ["Salbutamol", "Budesonide"],
    "Hypertension": ["Amlodipine", "Losartan"],
    "Migraine": ["Sumatriptan", "Propranolol"],
    "Cervical spondylosis": ["NSAIDs", "Physical Therapy"],
    "Paralysis (brain hemorrhage)": ["Rehabilitation Therapy", "Anticoagulants"],
    "Jaundice": ["Hepatoprotective Agents", "Vitamin K"],
    "Malaria": ["Chloroquine", "Artemether-Lumefantrine"],
    "Chicken pox": ["Acyclovir", "Calamine Lotion"],
    "Dengue": ["Paracetamol", "Fluid Replacement Therapy"],
    "Typhoid": ["Ciprofloxacin", "Azithromycin"],
    "Hepatitis A": ["Rest", "Balanced Diet"],
    "Hepatitis B": ["Entecavir", "Tenofovir"],
    "Hepatitis C": ["Sofosbuvir", "Ledipasvir"],
    "Hepatitis D": ["Interferon Alpha", "Antivirals"],
    "Hepatitis E": ["Rest", "Supportive Care"],
    "Alcoholic hepatitis": ["Prednisolone", "Pentoxifylline"],
    "Tuberculosis": ["Isoniazid", "Rifampin"],
    "Common Cold": ["Paracetamol", "Decongestants"],
    "Pneumonia": ["Amoxicillin", "Azithromycin"],
    "Dimorphic hemorrhoids (piles)": ["Topical Ointments", "Fiber Supplements"],
    "Heart attack": ["Aspirin", "Clopidogrel"],
    "Varicose veins": ["Compression Stockings", "Sclerotherapy"],
    "Hypothyroidism": ["Levothyroxine"],
    "Hyperthyroidism": ["Methimazole", "Propylthiouracil"],
    "Hypoglycemia": ["Glucose Tablets", "Dextrose"],
    "Osteoarthritis": ["Acetaminophen", "NSAIDs"],
    "Arthritis": ["Methotrexate", "Sulfasalazine"],
    "(vertigo) Paroxysmal Positional Vertigo": ["Epley Maneuver", "Meclizine"],
    "Acne": ["Benzoyl Peroxide", "Clindamycin"],
    "Urinary tract infection": ["Nitrofurantoin", "Trimethoprim-Sulfamethoxazole"],
    "Psoriasis": ["Topical Corticosteroids", "Calcipotriol"],
    "Impetigo": ["Mupirocin", "Retapamulin"]
}

# Menampilkan judul proyek dan informasi anggota proyek
st.markdown("<h3 style='text-align: center; color: black;'>Project Machine Learning : Prediksi Penyakit</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Universitas Bina Sarana Informatika</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Slipi</h3>", unsafe_allow_html=True)

# Menambahkan ruang pemisah antara informasi judul dan anggota proyek
st.markdown("</br>", unsafe_allow_html=True)

# Menampilkan informasi anggota proyek, nim, dan mata kuliah
st.markdown("<h6 style='text-align: left; color: black;'>Eka Tama Prasetya - 17225004</h6>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: left; color: black;'>Apriyanto Dwi Herlambang - 17225079</h6>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: left; color: black;'>Mata Kuliah : Pembelajaran Mesin</h6>", unsafe_allow_html=True)

# Menampilkan nama dosen pembimbing untuk proyek ini
st.markdown("<h6 style='text-align: left; color: black;'>Dosen Pembimbing : Mushliha, M.Si</h6>", unsafe_allow_html=True)

# Menambahkan ruang pemisah sebelum bagian lainnya
st.markdown("</br>", unsafe_allow_html=True)

# Menanyakan kepada pengguna berapa banyak gejala yang dialami
# Fungsi st.number_input digunakan untuk menerima input berupa angka
# 'min_value' menentukan nilai minimal (1 gejala),
# 'max_value' menentukan nilai maksimal (jumlah gejala yang tersedia dalam list 'symptoms'),
# 'step' mengatur langkah kenaikan angka input (setiap langkah adalah 1),
# dan 'value' memberikan nilai default (1 gejala).
num_symptoms = st.number_input('Berapa banyak gejala yang Anda alami?', min_value=1, max_value=len(symptoms), step=1, value=1)

# Mengizinkan pengguna untuk memilih gejala berdasarkan jumlah yang telah mereka masukkan sebelumnya
# Pada setiap iterasi, aplikasi akan menampilkan dropdown untuk memilih gejala
# Fungsi st.selectbox digunakan untuk menampilkan pilihan gejala dari list 'symptoms'
# Pengguna dapat memilih gejala yang mereka alami dari dropdown yang tersedia
# 'key' digunakan untuk memastikan setiap selectbox memiliki identifikasi unik sesuai urutan gejala yang dipilih
selected_symptoms = []
for i in range(num_symptoms):
    symptom = st.selectbox(f'Gejala Ke - {i+1}', symptoms, key=f'symptom_{i}')
    selected_symptoms.append(symptom)

# Mengonversi gejala yang dipilih oleh pengguna menjadi vektor input 
# dimana 1 berarti gejala tersebut dipilih dan 0 berarti tidak dipilih,
# sesuai dengan urutan gejala dalam daftar 'symptoms'.
input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]

# Menyediakan tombol untuk memulai prediksi penyakit berdasarkan gejala yang dipilih oleh pengguna
# Ketika tombol ditekan, program akan:
# - Mengonversi data input gejala menjadi format yang cocok untuk model prediksi
# - Melakukan prediksi penyakit dengan model yang sudah dilatih
# - Menampilkan hasil prediksi penyakit
# - Menampilkan rekomendasi obat berdasarkan penyakit yang diprediksi

if st.button("Prediksi Penyakit"):
    try:
        # Mengubah input gejala menjadi format array 2D untuk dimasukkan ke dalam model
        input_data = np.array(input_vector).reshape(1, -1)
        
        # Memanfaatkan model untuk memprediksi penyakit
        prediction = model.predict(input_data)
        predicted_disease = label_encoder.inverse_transform(prediction)[0]  # Mengembalikan nama penyakit dari kode prediksi
        
        # Menampilkan hasil prediksi penyakit
        st.write(f'Prediksi Penyakit: **{predicted_disease}**')
        
        # Menampilkan rekomendasi obat yang sesuai jika tersedia
        if predicted_disease in medication_db:
            drugs = medication_db[predicted_disease]
            st.write("Rekomendasi Obat:")
            for drug in drugs:
                st.write(f"- {drug}")
        else:
            st.write("No specific drug recommendations available for this condition.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")  # Menampilkan pesan error jika terjadi kesalahan selama prediksi
