import streamlit as st
import pandas as pd
import joblib

# ---------------- Load saved models & preprocessors ---------------- #
risk_model = joblib.load("anova_superensemble_model.pkl")
risk_scaler = joblib.load("anova_scaler.pkl")
risk_label_encoder = joblib.load("anova_labelencoder.pkl")

hr_model = joblib.load("targethr_randomforest_model.pkl")
hr_scaler = joblib.load("targethr_randomforest_scaler.pkl")
hr_label_encoder = joblib.load("targethr_labelencoder.pkl")

duration_model = joblib.load("duration_xgboost_model.pkl")
duration_scaler = joblib.load("duration_xgboost_scaler.pkl")
duration_label_encoder = joblib.load("duration_xgboost_labelencoder.pkl")

# ---------------- Streamlit App Title & Intro ---------------- #
st.set_page_config(page_title="Personalized Exercise Prescription", page_icon="🫀")

# Replace st.title(...) with this customized single-line title
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
        <span style="font-size:40px;">🏥</span>
        <h1 style="margin:0;padding:0;white-space:nowrap;">Personalized Exercise Prescription</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<div style='background-color:#f1f8e9;padding:20px;border-radius:10px;margin-bottom:20px;'>
    <h4>Welcome to your personalized cardiac exercise prescription app! 🎯</h4>
    <p>Fill in the patient's information below, then click <b>Predict Personalized Exercise Plan</b> to get a recommended risk level, target heart rate, and exercise duration.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- UI: Demographics & Clinical Inputs ---------------- #
st.subheader("👤 Patient Profile & Assessments")
age = st.number_input("Age", min_value=18, max_value=120, value=50)
gender = st.selectbox("Gender", ["M", "F"])
occupation = st.selectbox("Occupation", ["employed", "self-employed", "retired", "not working"])
marital_status = st.selectbox("Marital Status", ["single", "married", "divorced", "widow"])
lives_with = st.selectbox("Lives With", ["alone", "family", "partner"])

st.subheader("🩺 Medical History / Risk Factors")
smoking = st.selectbox("Smoking", ["no", "yes", "ex-smoker"])
family_history = st.selectbox("Family History", ["yes", "no"])
riskfactor_hpt = st.selectbox("Risk Factor - HPT", ["yes", "no"])
riskfactor_dm = st.selectbox("Risk Factor - DM", ["yes", "no"])
riskfactor_hpl = st.selectbox("Risk Factor - HPL", ["yes", "no"])
riskfactor_exercise = st.selectbox("Risk Factor - Exercise", ["inactive", "moderate", "active"])
riskfactor_stress = st.selectbox("Risk Factor - Stress", ["yes", "no"])
riskfactor_bmi = st.selectbox("Risk Factor - BMI", ["underweight", "healthy", "overweight", "obese"])
riskfactor_echo_ef = st.selectbox("Risk Factor - ECHO - EF", ["normal", "borderline", "reduced"])

st.subheader("🏃 Exercise Habits")
exercise_mode = st.selectbox("Exercise Habit - Mode", ["walking", "jogging", "cycling", "no", "others"])
exercise_frequency = st.number_input("Exercise Habit - Frequency (per week)", min_value=0, max_value=14, value=3)
exercise_duration = st.number_input("Exercise Habit - Duration (minutes)", min_value=0, max_value=300, value=30)

st.subheader("🧪 Clinical Test Results")
testtoday_mets = st.selectbox("Test Today - METS", ["low intensity", "moderate intensity", "high intensity"])
testtoday_terminationcause = st.selectbox("Test Today - Termination Cause", ["Complete Test", "Fatigue", "Medical Condition", "Physical Discomfort"])
testtoday_peakhr = st.selectbox("Test Today - peak HR", ["low intensity", "moderate intensity", "high intensity", "maximum intensity", "above maximum intensity"])
ecgresting = st.selectbox("ECG Resting", ["normal", "sinus rhythm", "T wave inversion", "ST depression", "Q wave", "ectopics"])
diagnosis = st.selectbox("Diagnosis", ["PCI", "CABG", "conservative", "surgical"])

st.subheader("🦵 Functional & Muscle Assessments")
rom = st.selectbox("ROM", ["normal", "abnormal"])

# ---------------- Prediction Button ---------------- #
if st.button("🔎 Predict Personalized Exercise Plan"):
    # ---------------- Build DataFrame from inputs ---------------- #
    input_dict = {
        'Age': age,
        'Gender': gender,
        'Occupation': occupation,
        'MaritalStatus': marital_status,
        'LivesWith': lives_with,
        'Smoking': smoking,
        'FamilyHistory': family_history,
        'RiskFactor-HPT': riskfactor_hpt,
        'RiskFactor-DM': riskfactor_dm,
        'RiskFactor-HPL': riskfactor_hpl,
        'RiskFactor-Exercise': riskfactor_exercise,
        'RiskFactor-Stress': riskfactor_stress,
        'RiskFactor-BMI': riskfactor_bmi,
        'RiskFactor-ECHO-EF': riskfactor_echo_ef,
        'ExerciseHabit-Mode': exercise_mode,
        'ExerciseHabit-Frequency': exercise_frequency,
        'ExerciseHabit-Duration': exercise_duration,
        'TestToday-METS': testtoday_mets,
        'TestToday-TerminationCause': testtoday_terminationcause,
        'TestToday-peakHR': testtoday_peakhr,
        'ECGResting': ecgresting,
        'Diagnosis': diagnosis,
        'ROM': rom,
        # ➕ Add other features you trained on, if needed
    }
    input_df = pd.DataFrame([input_dict])
    st.write("🔎 Input Data Preview:", input_df)

   # ---------------- Risk Level Prediction ---------------- #
    risk_input_encoded = pd.get_dummies(input_df)
    risk_input_encoded = risk_input_encoded.reindex(columns=risk_scaler.feature_names_in_, fill_value=0)
    risk_input_scaled = risk_scaler.transform(risk_input_encoded)
    risk_pred = risk_model.predict(risk_input_scaled)
    risk_label = risk_label_encoder.inverse_transform(risk_pred)[0]

    # Optional probability display (if model supports predict_proba)
    try:
        risk_probs = risk_model.predict_proba(risk_input_scaled)
        risk_confidence = f" ({np.max(risk_probs[0])*100:.1f}%)"
    except AttributeError:
        risk_confidence = ""

    st.markdown(
        f"<div style='background-color:#e8f5e9;padding:15px;border-radius:10px;'>"
        f"<h4>📝 Predicted Risk Level: <span style='color:#2e7d32;'>{risk_label}{risk_confidence}</span></h4>"
        f"</div>", unsafe_allow_html=True
    )

    # Add predicted RiskLevel into DataFrame for next stages
    input_df['RiskLevel'] = risk_label

    # ---------------- Target HR Prediction ---------------- #
    hr_input_encoded = pd.get_dummies(input_df)
    hr_input_encoded = hr_input_encoded.reindex(columns=hr_scaler.feature_names_in_, fill_value=0)
    hr_input_scaled = hr_scaler.transform(hr_input_encoded)
    hr_pred = hr_model.predict(hr_input_scaled)
    hr_label = hr_label_encoder.inverse_transform(hr_pred)[0]

    st.markdown(
        f"<div style='background-color:#e3f2fd;padding:15px;border-radius:10px;'>"
        f"<h4>❤️ Target Heart Rate: <span style='color:#1565c0;'>{hr_label} bpm</span></h4>"
        f"</div>", unsafe_allow_html=True
    )

    # ---------------- Duration Prediction ---------------- #
    duration_input_encoded = pd.get_dummies(input_df)
    duration_input_encoded = duration_input_encoded.reindex(columns=duration_scaler.feature_names_in_, fill_value=0)
    duration_input_scaled = duration_scaler.transform(duration_input_encoded)
    duration_pred = duration_model.predict(duration_input_scaled)
    duration_label = duration_label_encoder.inverse_transform(duration_pred)[0]

    st.markdown(
        f"<div style='background-color:#e8eaf6;padding:15px;border-radius:10px;'>"
        f"<h4>⏱️ Recommended Exercise Duration: <span style='color:#4527a0;'>{duration_label} minutes</span></h4>"
        f"</div>", unsafe_allow_html=True
    )