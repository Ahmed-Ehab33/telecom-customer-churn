import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #12121f 100%);
    border-right: 1px solid #1e1e35;
}
[data-testid="stSidebar"] * { color: #c8c8e0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #7878aa !important; font-size: 0.75rem !important; letter-spacing: 0.08em; text-transform: uppercase; }

/* ── Main background ── */
.main .block-container { padding: 2rem 3rem; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #0d0d1f 0%, #12082a 50%, #0a1628 100%);
    border: 1px solid #1e1e40;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, #5533ff22 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 160px; height: 160px;
    background: radial-gradient(circle, #00d4ff18 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #5533ff;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ffffff 0%, #a090ff 60%, #00d4ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
}
.hero p {
    color: #6868a0;
    font-size: 1rem;
    margin: 0;
}

/* ── Cards ── */
.card {
    background: #0f0f1e;
    border: 1px solid #1e1e38;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #4444aa;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.card-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8e8ff;
}

/* ── Risk badge ── */
.risk-high {
    background: linear-gradient(135deg, #2a0a0a, #1a0505);
    border: 1px solid #ff3b3b44;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, #0a1a10, #050f08);
    border: 1px solid #00ff8844;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    text-align: center;
}
.risk-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.risk-title-high { font-size: 1.3rem; font-weight: 700; color: #ff6b6b; }
.risk-title-low  { font-size: 1.3rem; font-weight: 700; color: #00ff88; }
.risk-sub { font-size: 0.85rem; color: #6868a0; margin-top: 0.3rem; }

/* ── Probability bar ── */
.prob-bar-bg {
    background: #1a1a30;
    border-radius: 100px;
    height: 10px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.prob-bar-fill-high {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #ff3b3b, #ff8c00);
    transition: width 0.8s ease;
}
.prob-bar-fill-low {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #00c96e, #00d4ff);
    transition: width 0.8s ease;
}

/* ── Summary table ── */
.summary-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #1a1a30;
    font-size: 0.9rem;
}
.summary-row:last-child { border-bottom: none; }
.summary-key { color: #5555a0; font-family: 'Space Mono', monospace; font-size: 0.75rem; }
.summary-val { color: #e0e0ff; font-weight: 600; }

/* ── Sidebar title ── */
.sidebar-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: #5533ff;
    text-transform: uppercase;
    padding: 1rem 0 0.5rem 0;
    border-top: 1px solid #1e1e35;
    margin-top: 0.5rem;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #5533ff, #3311dd) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6644ff, #4422ee) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px #5533ff44 !important;
}

/* ── Info box ── */
.info-box {
    background: #0d0d20;
    border: 1px solid #1e1e38;
    border-left: 3px solid #5533ff;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    font-size: 0.9rem;
    color: #8888bb;
}

/* ── Selectbox / slider overrides ── */
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: #0d0d1e !important;
    border: 1px solid #2a2a50 !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("xgb_churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">📡 AI-Powered Analytics</div>
    <h1>ChurnSense</h1>
    <p>Predict customer churn probability in real-time using XGBoost machine learning</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">👤 Demographics</div>', unsafe_allow_html=True)
    gender         = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Partner", ["Yes", "No"])
    dependents     = st.selectbox("Dependents", ["Yes", "No"])

    st.markdown('<div class="sidebar-title">📋 Account</div>', unsafe_allow_html=True)
    tenure             = st.slider("Tenure (months)", 0, 72, 12)
    contract           = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing  = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method     = st.selectbox("Payment Method",
                            ["Credit card (automatic)", "Electronic check",
                             "Mailed check", "Bank transfer (automatic)"])

    st.markdown('<div class="sidebar-title">💳 Charges</div>', unsafe_allow_html=True)
    monthly_charges = st.slider("Monthly Charges ($)", 0, 150, 50)
    total_charges   = st.slider("Total Charges ($)", 0, 10000, 1000)

    st.markdown('<div class="sidebar-title">📶 Services</div>', unsafe_allow_html=True)
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Churn", use_container_width=True)


# ── Main content ───────────────────────────────────────────────────────────────
if predict_btn:

    # Build input
    input_data = pd.DataFrame({
        'seniorcitizen':                         [1 if senior_citizen == "Yes" else 0],
        'tenure':                                [tenure],
        'monthlycharges':                        [monthly_charges],
        'totalcharges':                          [total_charges],
        'gender_Male':                           [1 if gender == "Male" else 0],
        'partner_Yes':                           [1 if partner == "Yes" else 0],
        'dependents_Yes':                        [1 if dependents == "Yes" else 0],
        'multiplelines_No phone service':        [1 if multiple_lines == "No phone service" else 0],
        'multiplelines_Yes':                     [1 if multiple_lines == "Yes" else 0],
        'contract_One year':                     [1 if contract == "One year" else 0],
        'contract_Two year':                     [1 if contract == "Two year" else 0],
        'paperlessbilling_Yes':                  [1 if paperless_billing == "Yes" else 0],
        'paymentmethod_Credit card (automatic)': [1 if payment_method == "Credit card (automatic)" else 0],
        'paymentmethod_Electronic check':        [1 if payment_method == "Electronic check" else 0],
        'paymentmethod_Mailed check':            [1 if payment_method == "Mailed check" else 0],
    })

    numeric_cols = ['tenure', 'monthlycharges', 'totalcharges']
    input_scaled = input_data.copy()
    input_scaled[numeric_cols] = scaler.transform(input_data[numeric_cols])

    prediction  = int(model.predict(input_scaled)[0])
    probability = float(model.predict_proba(input_scaled)[0][1])
    pct         = f"{probability:.1%}"
    bar_width   = f"{probability*100:.1f}%"

    # ── Results layout ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

    with col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="risk-high">
                <div class="risk-icon">⚠️</div>
                <div class="risk-title-high">High Churn Risk</div>
                <div class="risk-sub">Immediate action recommended</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <div class="risk-icon">✅</div>
                <div class="risk-title-low">Low Churn Risk</div>
                <div class="risk-sub">Customer likely to stay</div>
            </div>""", unsafe_allow_html=True)

    with col2:
        fill_class = "prob-bar-fill-high" if prediction == 1 else "prob-bar-fill-low"
        color      = "#ff6b6b" if prediction == 1 else "#00ff88"
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Churn Probability</div>
            <div class="card-value" style="color:{color}; font-size:2.2rem;">{pct}</div>
            <div class="prob-bar-bg">
                <div class="{fill_class}" style="width:{bar_width}"></div>
            </div>
        </div>
        <div class="card">
            <div class="card-label">Tenure</div>
            <div class="card-value">{tenure} <span style="font-size:0.9rem;color:#5555a0">months</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Customer Summary</div>
            <div class="summary-row">
                <span class="summary-key">CONTRACT</span>
                <span class="summary-val">{contract}</span>
            </div>
            <div class="summary-row">
                <span class="summary-key">PAYMENT</span>
                <span class="summary-val">{payment_method}</span>
            </div>
            <div class="summary-row">
                <span class="summary-key">MONTHLY</span>
                <span class="summary-val">${monthly_charges}</span>
            </div>
            <div class="summary-row">
                <span class="summary-key">TOTAL</span>
                <span class="summary-val">${total_charges:,}</span>
            </div>
            <div class="summary-row">
                <span class="summary-key">GENDER</span>
                <span class="summary-val">{gender}</span>
            </div>
            <div class="summary-row">
                <span class="summary-key">PARTNER</span>
                <span class="summary-val">{partner}</span>
            </div>
            <div class="summary-row">
                <span class="summary-key">SENIOR</span>
                <span class="summary-val">{senior_citizen}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Insight bar ─────────────────────────────────────────────────────────────
    if prediction == 1:
        tips = []
        if contract == "Month-to-month":
            tips.append("month-to-month contract increases churn risk")
        if monthly_charges > 70:
            tips.append("high monthly charges detected")
        if tenure < 12:
            tips.append("low tenure — customer is relatively new")
        tip_text = " · ".join(tips) if tips else "review customer engagement history"
        st.markdown(f"""
        <div class="info-box" style="margin-top:1rem; border-left-color:#ff6b6b;">
            <strong style="color:#ff6b6b;">⚡ Risk Factors Detected:</strong>&nbsp;&nbsp;{tip_text}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box" style="margin-top:1rem;">
            <strong style="color:#00ff88;">💡 Insight:</strong>&nbsp;&nbsp;
            This customer shows strong retention signals. Consider upsell opportunities.
        </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="info-box">
        👈 &nbsp;Fill in customer details in the sidebar and click <strong style="color:#a090ff;">Predict Churn</strong> to get started.
    </div>
    """, unsafe_allow_html=True)

    # ── Placeholder stats ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="card">
            <div class="card-label">Model</div>
            <div class="card-value">XGBoost</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card">
            <div class="card-label">Features</div>
            <div class="card-value">15</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card">
            <div class="card-label">Threshold</div>
            <div class="card-value">0.35</div>
        </div>""", unsafe_allow_html=True)
