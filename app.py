import base64
import numpy as np
import streamlit as st
from joblib import load

import os

# Path ke gambar
image_path = "/images/background-streamlit.png"

# Encode gambar ke Base64
with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Gaya CSS dengan gambar Base64
background_image = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{encoded_string}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
[data-testid="stSidebar"] {{
    background: rgba(0,0,0,0); /* Membuat sidebar transparan */
}}
button[role="button"] {{
    color: white;
    background-color: black;
    border: 2px solid black;
    padding: 10px 20px;
    border-radius: 5px;
    font-size: 16px;
}}
button[role="button"]:hover {{
    color: black;
    background-color: white;
    border-color: black;
}}
button[role="button"]:active {{
    color: black;
    background-color: white;
    border-color: black;
}}
h1, h2, h3, h4, h5, h6 {{
    color: black;  /* For header styles */
}}
.stTextInput label, .stSelectbox label, .stNumberInput label {{
    color: black;  /* Color for number input, selectbox, and other input labels */
}}
.stMarkdown, .stWrite, .stText {{
    color: black;  /* Change text color of written output */
}}
</style>
"""

# Terapkan background ke Streamlit
st.markdown(background_image, unsafe_allow_html=True)

# Load the saved model and label encoder
try:
    model = load('diagnosis_predictor.joblib')
    label_encoder = load('label_encoder.joblib')
     # Ambil akurasi dan nama model dari metadata di objek model
    model_accuracy = model.score if hasattr(model, "score") else 0.93  # Jika score tidak ditemukan, gunakan default
    model_name = type(model).__name__
except Exception as e:
    st.error(f"Error loading model or label encoder: {e}")
    st.stop()

# Define the complete list of symptoms
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

# Create input fields
st.markdown("<h3 style='text-align: center; color: black;'>Project Machine Learning : Prediksi Penyakit</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Universitas Bina Sarana Informatika</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Slipi</h3>", unsafe_allow_html=True)
st.markdown("</br>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: left; color: black;'>Eka Tama Prasetya - 17225004</h6>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: left; color: black;'>Apriyanto Dwi Herlambang - 17225079</h6>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: left; color: black;'>Mata Kuliah : Pembelajaran Mesin", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: left; color: black;'>Dosen Pembimbing : Mushliha, M.Si", unsafe_allow_html=True)
st.markdown("</br>", unsafe_allow_html=True)

# Ask user for the number of symptoms
num_symptoms = st.number_input('Berapa banyak gejala yang Anda alami?', min_value=1, max_value=len(symptoms), step=1, value=1)

# Allow user to select symptoms based on the number they entered
selected_symptoms = []
for i in range(num_symptoms):
    symptom = st.selectbox(f'Gejala Ke - {i+1}', symptoms, key=f'symptom_{i}')
    selected_symptoms.append(symptom)

# Convert selected symptoms to input vector
input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]

# Predict the disease
if st.button("Prediksi Penyakit"):
    try:
        input_data = np.array(input_vector).reshape(1, -1)
        prediction = model.predict(input_data)
        predicted_disease = label_encoder.inverse_transform(prediction)[0]
        
        # Display the prediction
        st.write(f'Prediksi Penyakit: **{predicted_disease}**')
        
        # Display drug recommendations
        if predicted_disease in medication_db:
            drugs = medication_db[predicted_disease]
            st.write("Rekomendasi Obat:")
            for drug in drugs:
                st.write(f"- {drug}")
        else:
            st.write("No specific drug recommendations available for this condition.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
