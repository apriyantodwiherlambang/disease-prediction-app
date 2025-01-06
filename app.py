import numpy as np
import streamlit as st
from joblib import load

# Load the saved model and label encoder
try:
    model = load('diagnosis_predictor.joblib')
    label_encoder = load('label_encoder.joblib')
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


# Create input fields
st.markdown("<h2 style='text-align: center;'>Universitas Bina Sarana Informatika</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Prediksi Penyakit</h2>", unsafe_allow_html=True)

# Ask user for the number of symptoms
num_symptoms = st.number_input('How many symptoms do you have?', min_value=1, max_value=len(symptoms), step=1, value=1)

# Allow user to select symptoms based on the number they entered
selected_symptoms = []
for i in range(num_symptoms):
    symptom = st.selectbox(f'Select symptom {i+1}', symptoms, key=f'symptom_{i}')
    selected_symptoms.append(symptom)

# Convert selected symptoms to input vector
input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]

# Predict the disease
if st.button("Predict Diagnosis"):
    try:
        input_data = np.array(input_vector).reshape(1, -1)
        prediction = model.predict(input_data)
        predicted_disease = label_encoder.inverse_transform(prediction)[0]
        
        # Display the prediction
        st.write(f'Predicted Diagnosis: **{predicted_disease}**')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
