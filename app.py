import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapsed for a cleaner "app" feel
)

# --- MASSIVE UI OVERHAUL (DEEP CSS INJECTION) ---
st.markdown("""
<style>
    /* 1. Aurora Animated Background */
    .stApp {
        background:
            radial-gradient(circle at 20% 20%, rgba(0,255,255,0.18), transparent 25%),
            radial-gradient(circle at 80% 30%, rgba(138,43,226,0.22), transparent 25%),
            radial-gradient(circle at 40% 80%, rgba(0,191,255,0.18), transparent 25%),
            linear-gradient(
                -45deg,
                #050816,
                #0b1026,
                #101935,
                #050816
            ) !important;
        background-size: 300% 300% !important;
        animation: aurora 20s ease infinite !important;
    }
    
    @keyframes aurora {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Make Streamlit's default top-header transparent AND click-through */
    [data-testid="stHeader"] {
        background-color: transparent !important;
        pointer-events: none !important; /* Fix for unclickable header links! */
    }
    
    /* Push main content down so it isn't hidden by the fixed header */
    .block-container {
        padding-top: 100px !important;
    }

    /* 2. Glassmorphism for the 3 Input Columns */
    [data-testid="column"] {
        background: rgba(10, 15, 30, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 20px;
    }
    
    /* Interactive hover effect for the columns */
    [data-testid="column"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 255, 255, 0.1);
        border: 1px solid rgba(0, 255, 255, 0.3);
    }

    /* 3. White Glowing Title Animation */
    .glow-title {
        font-size: 4.5em;
        font-weight: 900;
        text-align: center;
        color: #ffffff; /* White text */
        animation: pulseGlow 4s ease-in-out infinite;
        margin-bottom: 0.1em;
    }
    
    @keyframes pulseGlow {
        0% { text-shadow: 0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(0,255,255,0.6), 0 0 40px rgba(0,255,255,0.4); }
        50% { text-shadow: 0 0 15px rgba(255,255,255,0.8), 0 0 30px rgba(138,43,226,0.8), 0 0 50px rgba(138,43,226,0.5); }
        100% { text-shadow: 0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(0,255,255,0.6), 0 0 40px rgba(0,255,255,0.4); }
    }

    /* 4. Subheader Custom Styling */
    h3 {
        color: #00ffff !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 10px;
        margin-bottom: 25px !important;
    }

    /* 5. Custom Inputs (Perfectly Uniform Dark Glass) */
    
    /* Target the base wrappers for both Select and Number/Text inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] {
        background-color: rgba(15, 23, 42, 0.7) !important; /* Uniform blue-dark glass */
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }

    /* Force the actual text field inside number_input to be transparent */
    div[data-baseweb="base-input"] > input {
        background-color: transparent !important;
        color: white !important;
        -webkit-text-fill-color: white !important;
    }

    /* Force all text elements inside selectbox to be transparent background & white text */
    div[data-baseweb="select"] * {
        background-color: transparent !important;
        color: white !important;
    }

    /* Fix number input up/down buttons */
    div[data-baseweb="input"] button {
        background-color: transparent !important;
        color: white !important;
    }
    div[data-baseweb="input"] button:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* 6. Primary Button Overhaul (Massive Glowing Pill matching Aurora) */
    [data-testid="baseButton-primary"] {
        background: linear-gradient(90deg, #00ffff, #8a2be2) !important;
        border: none !important;
        padding: 1.5rem !important;
        border-radius: 40px !important;
        font-size: 1.4rem !important;
        font-weight: 900 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(0, 255, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="baseButton-primary"]:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 8px 30px rgba(0, 255, 255, 0.6) !important;
    }

    /* 7. Results Cards */
    .result-card-danger {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.2), rgba(255, 0, 0, 0.1));
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 75, 75, 0.3);
        border-left: 8px solid #ff4b4b;
        padding: 25px;
        border-radius: 15px;
        color: white;
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 10px 30px rgba(255, 75, 75, 0.2);
    }
    
    .result-card-safe {
        background: linear-gradient(135deg, rgba(9, 171, 59, 0.2), rgba(0, 255, 0, 0.1));
        backdrop-filter: blur(15px);
        border: 1px solid rgba(9, 171, 59, 0.3);
        border-left: 8px solid #09ab3b;
        padding: 25px;
        border-radius: 15px;
        color: white;
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 10px 30px rgba(9, 171, 59, 0.2);
    }

    @keyframes slideUp {
        from { transform: translateY(40px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    /* 8. Sticky Header styling */
    .custom-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 40px;
        background: rgba(5, 8, 22, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        z-index: 999999 !important; /* Increased z-index */
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .header-logo {
        font-size: 1.5em;
        font-weight: 800;
        color: #fff;
        display: flex;
        align-items: center;
        gap: 10px;
        z-index: 2;
    }
    
    .header-title {
        color: #ffffff;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.6), 0 0 15px rgba(138, 43, 226, 0.4);
    }

    .header-links {
        display: flex;
        gap: 25px;
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1;
        pointer-events: auto; /* Ensure clicks register here */
    }
    
    .header-links a {
        color: #94a3b8;
        text-decoration: none;
        font-size: 1em;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    
    .header-links a:hover {
        color: #00ffff;
        transform: translateY(-2px);
    }

    /* Hide defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. PROFILE HEADER (STICKY)
# -----------------------------------------------------------------------------
# Profile header with custom links
st.markdown("""
<div class="custom-header">
    <div class="header-logo">
        🫀 <span class="header-title">Diagnostic Cardiology AI</span>
    </div>
    <div class="header-links">
        <a href="https://github.com/Chandu-gummadavelly" target="_blank">🐙 GitHub</a>
        <a href="https://www.linkedin.com/in/chandu-gummadavelly-524830330" target="_blank">💼 LinkedIn</a>
        <a href="https://github.com/Chandu-gummadavelly" target="_blank">🌐 Portfolio</a>
        <a href="mailto:chandugummadavelly@gmail.com">📧 Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. LOAD THE MODEL PIPELINE
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    # This loads the exact pipeline we built, complete with one-hot encoding!
    return joblib.load('heart_disease_pipeline.joblib')

try:
    pipeline = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Could not load the model. Ensure 'heart_disease_pipeline.joblib' is in the same directory. Error: {e}")

# -----------------------------------------------------------------------------
# 4. UI LAYOUT & INPUTS
# -----------------------------------------------------------------------------
st.markdown('<div class="glow-title">🫀 Diagnostic Cardiology AI</div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #cbd5e1; margin-bottom: 3rem; font-size: 1.2em; font-weight: 300;'>
    Enter the patient's clinical parameters below. The system will process the raw data 
    through our cost-complexity pruned Decision Tree to estimate the likelihood of heart disease.
</div>
""", unsafe_allow_html=True)

# Create columns for better UI layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Vitals & Demographics")
    age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120, step=1)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    st.markdown("### Clinical Symptoms")
    cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], format_func=lambda x: {
        1: "1: Typical Angina", 
        2: "2: Atypical Angina", 
        3: "3: Non-anginal Pain", 
        4: "4: Asymptomatic"
    }[x])
    exang = st.selectbox("Exercise Induced Angina?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150, step=1)

with col3:
    st.markdown("### ECG & Imaging")
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], format_func=lambda x: {
        0: "0: Normal",
        1: "1: ST-T Wave Abnormality",
        2: "2: Left Ventricular Hypertrophy"
    }[x])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[1, 2, 3], format_func=lambda x: {
        1: "1: Upsloping", 
        2: "2: Flat", 
        3: "3: Downsloping"
    }[x])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0.0, 1.0, 2.0, 3.0])
    thal = st.selectbox("Thalassemia Status", options=[3.0, 6.0, 7.0], format_func=lambda x: {
        3.0: "3: Normal", 
        6.0: "6: Fixed Defect", 
        7.0: "7: Reversable Defect"
    }[x])

st.markdown("<br><br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. PREDICTION LOGIC
# -----------------------------------------------------------------------------
# Center the predict button
col_btn1, col_btn2, col_btn3 = st.columns([1, 1.5, 1])

with col_btn2:
    predict_clicked = st.button("🔍 Run Diagnostic Analysis", use_container_width=True, type="primary")

if predict_clicked and model_loaded:
    # 1. Pack the raw inputs into a DataFrame matching the original data structure
    # NOTE: The keys must EXACTLY match the column names of X_train
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])
    
    # 2. Predict (Pipeline handles all median-imputation & OHE silently under the hood)
    prediction = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)[0]
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Display Results
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 1:
            st.markdown("""
            <div class="result-card-danger">
                <h2 style='margin-top: 0; color: #ff4b4b; font-weight: 800;'>🚨 HIGH RISK DETECTED</h2>
                <p style='font-size: 1.1em;'>The model identified clinical markers strongly correlating with heart disease. Immediate clinical review is recommended.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card-safe">
                <h2 style='margin-top: 0; color: #09ab3b; font-weight: 800;'>✅ LOW RISK</h2>
                <p style='font-size: 1.1em;'>The model did not find significant patterns of heart disease. Continue standard preventative care.</p>
            </div>
            """, unsafe_allow_html=True)
            
    with res_col2:
        st.markdown(f"""
        <div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1); color: white;">
            <h4 style="color: #cbd5e1; margin-bottom: 10px;">Diagnostic Confidence Score</h4>
            <h1 style="color: {'#ff4b4b' if prediction == 1 else '#09ab3b'}; font-size: 3.5em; margin: 0;">{probabilities[prediction] * 100:.1f}%</h1>
            <p style="color: #94a3b8; font-style: italic; margin-top: 5px;">Based on algorithmic tree path certainty.</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(probabilities[prediction]))
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("ℹ️ *Disclaimer: This application is for educational purposes and is not a substitute for professional medical advice.*")

# -----------------------------------------------------------------------------
# 6. CUSTOM FOOTER
# -----------------------------------------------------------------------------
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; color: #94a3b8; font-size: 0.95em; border-top: 1px solid rgba(255,255,255,0.05);">
    <p>Model built by <b style="color: #00ffff;">chandu_gummadavelly</b> as a project while learning.</p>
</div>
""", unsafe_allow_html=True)
