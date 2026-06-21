import streamlit as st
import pandas as pd
import joblib
import requests  # Using direct API requests to bypass ALL library bugs!

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STATE INITIALIZATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize chat state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Hi there! I am your AI Medical Assistant. I can explain complex medical terms, give you suggestions to lower your risk, or help you understand your results. How can I help?",
        }
    ]

# --- RESTORED STABLE UI (DEEP CSS INJECTION) ---
st.markdown(
    """
<style>
    /* 1. Aurora Animated Background */
    .stApp {
        background:
            radial-gradient(circle at 20% 20%, rgba(0,255,255,0.18), transparent 25%),
            radial-gradient(circle at 80% 30%, rgba(138,43,226,0.22), transparent 25%),
            radial-gradient(circle at 40% 80%, rgba(0,191,255,0.18), transparent 25%),
            linear-gradient(-45deg, #050816, #0b1026, #101935, #050816) !important;
        background-size: 300% 300% !important;
        animation: aurora 20s ease infinite !important;
    }
    
    @keyframes aurora {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    [data-testid="stHeader"] {
        background-color: transparent !important;
        pointer-events: none !important; 
    }
    
    .block-container {
        padding-top: 100px !important;
    }

    /* 2. Glassmorphism for the 3 Input Columns & Info Cards */
    [data-testid="column"], .info-card {
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
    
    [data-testid="column"]:hover, .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 255, 255, 0.1);
        border: 1px solid rgba(0, 255, 255, 0.3);
    }

    .info-card { color: #cbd5e1; line-height: 1.6; margin-bottom: 40px; }
    .info-card h4 {
        color: #00ffff; font-weight: 700; margin-top: 0; margin-bottom: 15px;
        border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px;
        display: flex; align-items: center; gap: 10px;
    }

    /* 3. White Glowing Title Animation */
    .glow-title {
        font-size: 4.5em; font-weight: 900; text-align: center; color: #ffffff; 
        animation: pulseGlow 4s ease-in-out infinite; margin-bottom: 0.1em;
    }
    
    @keyframes pulseGlow {
        0% { text-shadow: 0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(0,255,255,0.6), 0 0 40px rgba(0,255,255,0.4); }
        50% { text-shadow: 0 0 15px rgba(255,255,255,0.8), 0 0 30px rgba(138,43,226,0.8), 0 0 50px rgba(138,43,226,0.5); }
        100% { text-shadow: 0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(0,255,255,0.6), 0 0 40px rgba(0,255,255,0.4); }
    }

    /* 4. Subheader Custom Styling */
    h3 {
        color: #00ffff !important; font-weight: 600 !important; letter-spacing: 1px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 10px; margin-bottom: 25px !important;
    }

    /* 5. Custom Inputs */
    div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {
        background-color: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important; box-shadow: none !important;
    }
    div[data-baseweb="base-input"] > input {
        background-color: transparent !important; color: white !important; -webkit-text-fill-color: white !important;
    }
    div[data-baseweb="select"] * { background-color: transparent !important; color: white !important; }
    div[data-baseweb="input"] button { background-color: transparent !important; color: white !important; }
    div[data-baseweb="input"] button:hover { background-color: rgba(255, 255, 255, 0.1) !important; }

    /* 6. Primary Button Overhaul */
    [data-testid="baseButton-primary"] {
        background: linear-gradient(90deg, #00ffff, #8a2be2) !important; border: none !important;
        padding: 1.5rem !important; border-radius: 40px !important; font-size: 1.4rem !important;
        font-weight: 900 !important; letter-spacing: 2px !important; text-transform: uppercase !important;
        color: white !important; box-shadow: 0 4px 15px rgba(0, 255, 255, 0.3) !important; transition: all 0.3s ease !important;
    }
    [data-testid="baseButton-primary"]:hover {
        transform: scale(1.02) !important; box-shadow: 0 8px 30px rgba(0, 255, 255, 0.6) !important;
    }

    /* 7. Results Cards */
    .result-card-danger, .result-card-safe {
        backdrop-filter: blur(15px); padding: 25px; border-radius: 15px; color: white; animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .result-card-danger {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.2), rgba(255, 0, 0, 0.1));
        border: 1px solid rgba(255, 75, 75, 0.3); border-left: 8px solid #ff4b4b; box-shadow: 0 10px 30px rgba(255, 75, 75, 0.2);
    }
    .result-card-safe {
        background: linear-gradient(135deg, rgba(9, 171, 59, 0.2), rgba(0, 255, 0, 0.1));
        border: 1px solid rgba(9, 171, 59, 0.3); border-left: 8px solid #09ab3b; box-shadow: 0 10px 30px rgba(9, 171, 59, 0.2);
    }

    @keyframes slideUp { from { transform: translateY(40px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }

    /* 8. Sticky Header styling */
    .custom-header {
        position: fixed; top: 0; left: 0; right: 0; display: flex; justify-content: space-between; align-items: center;
        padding: 15px 40px; background: rgba(5, 8, 22, 0.85); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05); z-index: 999999 !important; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .header-logo { font-size: 1.5em; font-weight: 800; color: #fff; display: flex; align-items: center; gap: 10px; z-index: 2; }
    .header-title { color: #ffffff; text-shadow: 0 0 8px rgba(0, 255, 255, 0.6), 0 0 15px rgba(138, 43, 226, 0.4); }
    .header-links { display: flex; gap: 25px; position: absolute; left: 50%; transform: translateX(-50%); z-index: 1; pointer-events: auto; }
    .header-links a { color: #94a3b8; text-decoration: none; font-size: 1em; font-weight: 600; letter-spacing: 0.5px; transition: all 0.3s ease; }
    .header-links a:hover { color: #00ffff; transform: translateY(-2px); }

    /* -------------------------------------------------------------------------
       9. PREMIUM CHATBOT STYLING UPGRADE
    ------------------------------------------------------------------------- */
    
    /* Target the container wrapping the chat to make it a premium glass card */
    div[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stChatInput"]) {
        background: rgba(10, 15, 30, 0.6) !important;
        border: 1px solid rgba(0, 255, 255, 0.2) !important;
        border-radius: 20px !important;
        padding: 30px 20px !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
    }

    /* Individual Chat Bubbles */
    [data-testid="stChatMessage"] {
        background-color: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding: 20px !important;
        border-radius: 15px !important;
        margin-bottom: 20px !important;
        color: #f1f5f9;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
        font-size: 1.05em;
        line-height: 1.6;
    }
    
    /* Chat Avatars */
    [data-testid="stChatMessageAvatar"] {
        background-color: rgba(0, 255, 255, 0.1) !important;
        border: 1px solid rgba(0, 255, 255, 0.4) !important;
        color: #00ffff !important;
    }

    /* Chat Input Bar */
    [data-testid="stChatInput"] {
        background: rgba(5, 8, 22, 0.9) !important;
        border: 1px solid rgba(138, 43, 226, 0.4) !important;
        border-radius: 15px !important;
        padding: 5px 10px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
        margin-top: 15px !important;
    }
    
    [data-testid="stChatInput"]:focus-within {
        border: 1px solid rgba(0, 255, 255, 0.8) !important;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3) !important;
    }
    
    [data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        font-size: 1.05em !important;
    }
    
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. PROFILE HEADER (STICKY)
# -----------------------------------------------------------------------------
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 3. LOAD THE MODEL PIPELINE
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("heart_disease_pipeline.joblib")


try:
    pipeline = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(
        f"⚠️ Could not load the model. Ensure 'heart_disease_pipeline.joblib' is in the same directory. Error: {e}"
    )

# -----------------------------------------------------------------------------
# 4. HERO SECTION & MODEL EXPLANATION
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="glow-title">🫀 Diagnostic Cardiology AI</div>', unsafe_allow_html=True
)

st.markdown(
    """
<div style="position: relative; width: 100%; height: 250px; border-radius: 20px; overflow: hidden; margin-bottom: 2rem; box-shadow: 0 15px 35px rgba(0, 255, 255, 0.15); border: 1px solid rgba(255,255,255,0.05);">
<div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: linear-gradient(to right, rgba(5,8,22,1) 0%, rgba(5,8,22,0.3) 50%, rgba(5,8,22,1) 100%), linear-gradient(to top, rgba(5,8,22,1) 0%, transparent 100%); z-index: 1;"></div>
<img src="https://www.shutterstock.com/image-vector/health-care-medical-human-heart-260nw-2690054277.jpg" style="width: 100%; height: 100%; object-fit: cover; filter: hue-rotate(190deg) saturate(120%) brightness(0.8);">
<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 2; width: 90%; text-align: center;">
<h2 style="color: #ffffff; font-weight: 800; font-size: 2.2em; text-shadow: 0 4px 20px rgba(0,0,0,0.8); margin-bottom: 10px;">Precision Algorithm. Clinical Confidence.</h2>
<p style="color: #cbd5e1; font-size: 1.2em; font-weight: 400; text-shadow: 0 2px 10px rgba(0,0,0,0.8);">Advanced diagnostic insights powered by a rigorously pruned Cost-Complexity Decision Tree.</p>
</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px;">
<div class="info-card" style="flex: 1; min-width: 300px; margin-bottom: 0;">
<h4>🧠 The Algorithm</h4>
<p>This system operates on a <b>Cost-Complexity Pruned Decision Tree</b> trained on the UCI Cleveland dataset. Standard decision trees often memorize data (overfitting), making them dangerous and unreliable for medical use.</p>
<p>By applying mathematically rigorous <i>alpha-pruning</i> and K-Fold Cross-Validation, we strategically amputated redundant logic branches. The result is a mathematically optimal diagnostic path that maximizes out-of-sample accuracy while avoiding overfitting.</p>
</div>
<div class="info-card" style="flex: 1.5; min-width: 300px; margin-bottom: 0;">
<h4>🔬 Clinical Parameters description </h4>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
<div>
  <b style="color: #fff;">Age:</b><br>
  <span style="font-size: 0.9em;">Age of the patient in years.</span>
</div>

<div>
  <b style="color: #fff;">Chest Pain Type (CP):</b><br>
  <span style="font-size: 0.9em;">Ranges from Typical Angina, Atypical Angina, Non-Anginal Pain, to Asymptomatic.</span>
</div>

<div>
  <b style="color: #fff;">Resting ECG Results:</b><br>
  <span style="font-size: 0.9em;">Electrocardiogram results at rest (Normal, ST-T wave abnormality, or Left Ventricular Hypertrophy).</span>
</div>

<div>
  <b style="color: #fff;">Sex:</b><br>
  <span style="font-size: 0.9em;">Biological sex of the patient (Male or Female).</span>
</div>

<div>
  <b style="color: #fff;">Exercise-Induced Angina:</b><br>
  <span style="font-size: 0.9em;">Indicates whether chest pain occurs during exercise (Yes or No).</span>
</div>

<div>
  <b style="color: #fff;">ST Depression (Oldpeak):</b><br>
  <span style="font-size: 0.9em;">Amount of ST depression induced by exercise relative to rest.</span>
</div>

<div>
  <b style="color: #fff;">Resting Blood Pressure:</b><br>
  <span style="font-size: 0.9em;">Blood pressure measured at rest (in mm Hg).</span>
</div>

<div>
  <b style="color: #fff;">Maximum Heart Rate Achieved:</b><br>
  <span style="font-size: 0.9em;">Highest heart rate reached during exercise testing.</span>
</div>

<div>
  <b style="color: #fff;">Slope of Peak Exercise ST Segment:</b><br>
  <span style="font-size: 0.9em;">Describes the slope of the ST segment during peak exercise (Upsloping, Flat, or Downsloping).</span>
</div>

<div>
  <b style="color: #fff;">Serum Cholesterol:</b><br>
  <span style="font-size: 0.9em;">Cholesterol level in blood measured in mg/dL.</span>
</div>

<div>
  <b style="color: #fff;">Major Vessels Colored by Fluoroscopy (CA):</b><br>
  <span style="font-size: 0.9em;">Number of major blood vessels (0–3) visualized using fluoroscopy.</span>
</div>

<div>
  <b style="color: #fff;">Fasting Blood Sugar:</b><br>
  <span style="font-size: 0.9em;">Indicates whether fasting blood sugar is greater than 120 mg/dL (Yes or No).</span>
</div>

<div>
  <b style="color: #fff;">Thalassemia Status:</b><br>
  <span style="font-size: 0.9em;">Blood disorder status (Normal, Fixed Defect, or Reversible Defect).</span>
</div>
</div>
</div>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

# -----------------------------------------------------------------------------
# 5. UI LAYOUT & INPUTS
# -----------------------------------------------------------------------------
st.markdown(
    """<div style='text-align: center; color: #cbd5e1; margin-bottom: 3rem; font-size: 1.2em; font-weight: 300;'>Enter the patient's clinical parameters below to initialize the diagnostic sequence.</div>""",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Vitals & Demographics")
    age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
    sex = st.selectbox(
        "Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female"
    )
    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120, step=1
    )
    chol = st.number_input(
        "Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1
    )
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl?",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )

with col2:
    st.markdown("### Clinical Symptoms")
    cp = st.selectbox(
        "Chest Pain Type",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1: Typical Angina",
            2: "2: Atypical Angina",
            3: "3: Non-anginal Pain",
            4: "4: Asymptomatic",
        }[x],
    )
    exang = st.selectbox(
        "Exercise Induced Angina?",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    thalach = st.number_input(
        "Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150, step=1
    )

with col3:
    st.markdown("### ECG & Imaging")
    restecg = st.selectbox(
        "Resting ECG Results",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0: Normal",
            1: "1: ST-T Wave Abnormality",
            2: "2: Left Ventricular Hypertrophy",
        }[x],
    )
    oldpeak = st.number_input(
        "ST Depression Induced by Exercise",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
    )
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1: Upsloping", 2: "2: Flat", 3: "3: Downsloping"}[x],
    )
    ca = st.selectbox(
        "Number of Major Vessels Colored by Fluoroscopy", options=[0.0, 1.0, 2.0, 3.0]
    )
    thal = st.selectbox(
        "Thalassemia Status",
        options=[3.0, 6.0, 7.0],
        format_func=lambda x: {
            3.0: "3: Normal",
            6.0: "6: Fixed Defect",
            7.0: "7: Reversable Defect",
        }[x],
    )

st.markdown("<br><br>", unsafe_allow_html=True)

# Initialize prediction state variables
pred_status = "Not run yet"
pred_confidence = ""

# -----------------------------------------------------------------------------
# 6. PREDICTION LOGIC
# -----------------------------------------------------------------------------
col_btn1, col_btn2, col_btn3 = st.columns([1, 1.5, 1])

with col_btn2:
    predict_clicked = st.button(
        "🔍 Run Diagnostic Analysis", use_container_width=True, type="primary"
    )

if predict_clicked and model_loaded:
    input_data = pd.DataFrame(
        [
            {
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
                "thal": thal,
            }
        ]
    )

    prediction = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)[0]

    pred_status = "HIGH RISK" if prediction == 1 else "LOW RISK"
    pred_confidence = f"{probabilities[prediction] * 100:.1f}%"

    st.markdown("<br>", unsafe_allow_html=True)
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if prediction == 1:
            st.markdown(
                """<div class="result-card-danger"><h2 style='margin-top: 0; color: #ff4b4b; font-weight: 800;'>🚨 HIGH RISK DETECTED</h2><p style='font-size: 1.1em;'>The model identified clinical markers strongly correlating with heart disease. Immediate clinical review is recommended.</p></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="result-card-safe"><h2 style='margin-top: 0; color: #09ab3b; font-weight: 800;'>✅ LOW RISK</h2><p style='font-size: 1.1em;'>The model did not find significant patterns of heart disease. Continue standard preventative care.</p></div>""",
                unsafe_allow_html=True,
            )

    with res_col2:
        st.markdown(
            f"""<div style="background: rgba(0,0,0,0.3); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1); color: white;"><h4 style="color: #cbd5e1; margin-bottom: 10px;">Diagnostic Confidence Score</h4><h1 style="color: {'#ff4b4b' if prediction == 1 else '#09ab3b'}; font-size: 3.5em; margin: 0;">{pred_confidence}</h1><p style="color: #94a3b8; font-style: italic; margin-top: 5px;">Based on algorithmic tree path certainty.</p></div>""",
            unsafe_allow_html=True,
        )
        st.progress(float(probabilities[prediction]))

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "ℹ️ *Disclaimer: This application is for educational purposes and is not a substitute for professional medical advice.*"
    )

# -----------------------------------------------------------------------------
# 7. CENTERED AI ASSISTANT CARD (NO EXTERNAL LIBRARIES REQUIRED)
# -----------------------------------------------------------------------------
st.markdown(
    "<br><hr style='border-color: rgba(255,255,255,0.1); margin: 40px 0;'>",
    unsafe_allow_html=True,
)

# Using a wider layout [1, 4, 1] to give the chat room to breathe
chat_spacer1, chat_col, chat_spacer2 = st.columns([1, 4, 1])

with chat_col:
    st.markdown(
        """
    <h2 style="color: #00ffff; text-align: center; margin-bottom: 5px; font-weight: 800; text-shadow: 0 0 15px rgba(0,255,255,0.4);">🤖 AI Medical Assistant</h2>
    <p style="color: #cbd5e1; text-align: center; font-size: 1.1em; margin-bottom: 20px;">Your personal cardiology consultant. Ask about your inputs, results, or general heart health.</p>
    """,
        unsafe_allow_html=True,
    )

    # Increased height to 600px for a more substantial feel
    chat_container = st.container(height=600, border=True)

    with chat_container:
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input
        if prompt := st.chat_input("Ask a medical question..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.rerun()

        # The AI Logic
        if st.session_state.chat_history[-1]["role"] == "user":
            prompt = st.session_state.chat_history[-1]["content"]

            # Automatically grab Groq API key from Streamlit secrets
            api_key = st.secrets.get("GROQ_API_KEY")

            if not api_key:
                with st.chat_message("assistant"):
                    st.error(
                        "Please add GROQ_API_KEY to your .streamlit/secrets.toml file. You can get a free key at https://console.groq.com"
                    )
            else:
                try:
                    # DIRECT API REQUEST: Using Groq (OpenAI-compatible format)
                    system_prompt = f"""
                    You are a highly intelligent, empathetic, and professional AI Medical Assistant integrated into a Heart Disease Prediction application. 
                    
                    Here is the LIVE data currently entered by the patient into the application:
                    - Age: {age}
                    - Sex: {"Male" if sex == 1 else "Female"}
                    - Chest Pain Type: {cp}
                    - Resting Blood Pressure: {trestbps} mm Hg
                    - Serum Cholesterol: {chol} mg/dl
                    - Fasting Blood Sugar > 120: {"Yes" if fbs == 1 else "No"}
                    - Resting ECG: {restecg}
                    - Max Heart Rate Achieved: {thalach}
                    - Exercise Induced Angina: {"Yes" if exang == 1 else "No"}
                    - ST Depression: {oldpeak}
                    - Slope of Peak Exercise ST Segment: {slope}
                    - Major Vessels Colored (Fluoroscopy): {ca}
                    - Thalassemia Status: {thal}
                    
                    The patient's current Diagnostic Prediction Status is: {pred_status} (Confidence: {pred_confidence}).
                    
                    INSTRUCTIONS & STRICT BOUNDARIES:
                    1. STRICT DOMAIN EXPERTISE: You are strictly a medical AI assistant for this specific Heart Disease Prediction application.
                    2. REFUSE UNRELATED TOPICS: If the user asks about anything unrelated to cardiology, heart health, their clinical parameters, or their diagnostic results (for example: asking for recipes, celebrity information, coding help, general knowledge, or completely random questions), you MUST politely decline. Respond with something like: "I am a dedicated medical assistant for this application. I can only answer questions related to heart health, your clinical parameters, and your diagnostic results."
                    3. Answer the user's relevant medical/app questions clearly and concisely.
                    4. If the user asks about a parameter they inputted, explain it simply.
                    5. If their prediction status is "HIGH RISK" and they ask for advice, gently but firmly suggest they consult a healthcare professional. You may suggest looking for nearby diagnostic centers, cardiologists, and provide general lifestyle suggestions.
                    6. Always include a standard medical disclaimer when giving health advice.
                    """

                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        message_placeholder.markdown("🧠 Analyzing...")

                        api_url = "https://api.groq.com/openai/v1/chat/completions"
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        }

                        # Give the chatbot memory of the conversation
                        messages = [{"role": "system", "content": system_prompt}]
                        for msg in st.session_state.chat_history[:-1]:
                            messages.append(
                                {"role": msg["role"], "content": msg["content"]}
                            )
                        messages.append({"role": "user", "content": prompt})

                        payload = {
                            "model": "llama-3.1-8b-instant",
                            "messages": messages,
                        }

                        response = requests.post(api_url, headers=headers, json=payload)

                        if response.status_code == 200:
                            ai_text = response.json()["choices"][0]["message"][
                                "content"
                            ]
                            message_placeholder.markdown(ai_text)
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": ai_text}
                            )
                        else:
                            st.error(
                                f"Server Error {response.status_code}: {response.text}"
                            )

                except Exception as e:
                    with st.chat_message("assistant"):
                        st.error(f"Error connecting to AI: {e}")

# -----------------------------------------------------------------------------
# 8. CUSTOM FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    """
<div style="text-align: center; margin-top: 50px; padding: 20px; color: #94a3b8; font-size: 0.95em; border-top: 1px solid rgba(255,255,255,0.05);">
<p>Model built by <b style="color: #00ffff;">chandu_gummadavelly</b> as a project while learning.</p>
</div>
""",
    unsafe_allow_html=True,
)
