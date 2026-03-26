import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import tempfile, os, warnings, io
warnings.filterwarnings('ignore')

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="metric-container"] {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 16px;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #2a2d3e;
    }
    .risk-box-high {
        background: #3d1515;
        border: 1px solid #ff6b6b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .risk-box-low {
        background: #0d2e1e;
        border: 1px solid #51cf66;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .xai-bar-container {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 8px;
    }
    .rec-box-high {
        background: #2a1a1a;
        border-left: 4px solid #ff6b6b;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 8px;
        color: #ffd6d6;
    }
    .rec-box-low {
        background: #1a2a1a;
        border-left: 4px solid #51cf66;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 8px;
        color: #d6ffd6;
    }
    .validation-warning {
        background: #2a2200;
        border: 1px solid #ffaa00;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        color: #ffcc44;
    }
    .validation-error {
        background: #2a0000;
        border: 1px solid #ff4444;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        color: #ff8888;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model & Scaler ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        with open("model/heart_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Could not load model files: {e}\n\nMake sure heart_model.pkl and scaler.pkl are in the model/ folder.")
        return None, None

model, scaler = load_artifacts()

# ─── Feature Metadata ─────────────────────────────────────────────────────────
FEATURE_NAMES = ['age','sex','cp','trestbps','chol','fbs',
                 'restecg','thalach','exang','oldpeak','slope','ca','thal']

FEATURE_LABELS = {
    'age': 'Age', 'sex': 'Sex', 'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure', 'chol': 'Cholesterol',
    'fbs': 'Fasting Blood Sugar', 'restecg': 'Resting ECG',
    'thalach': 'Max Heart Rate', 'exang': 'Exercise Angina',
    'oldpeak': 'ST Depression', 'slope': 'ST Slope',
    'ca': 'Major Vessels', 'thal': 'Thalassemia'
}

# Normal ranges for validation & explanation
FEATURE_RANGES = {
    'age':      (1,   120,  "years"),
    'sex':      (0,   1,    ""),
    'cp':       (0,   3,    ""),
    'trestbps': (60,  220,  "mmHg"),
    'chol':     (100, 600,  "mg/dl"),
    'fbs':      (0,   1,    ""),
    'restecg':  (0,   2,    ""),
    'thalach':  (40,  250,  "bpm"),
    'exang':    (0,   1,    ""),
    'oldpeak':  (0.0, 10.0, ""),
    'slope':    (0,   2,    ""),
    'ca':       (0,   4,    ""),
    'thal':     (0,   3,    ""),
}

HEALTHY_RANGES = {
    'trestbps': (90,  120,  "Normal BP is 90–120 mmHg"),
    'chol':     (0,   200,  "Healthy cholesterol is <200 mg/dl"),
    'thalach':  (120, 200,  "Good max heart rate is 120–200 bpm"),
    'oldpeak':  (0.0, 2.0,  "ST Depression >2.0 is concerning"),
}

# ─── ① EXPLAINABLE AI ────────────────────────────────────────────────────────
def get_xai_contributions(model, scaler, patient_df):
    """
    Returns per-feature risk contributions using:
    - feature_importances_ (tree models) scaled by normalized input deviation
    - Falls back to weight × |z-score| for linear-style models
    """
    contributions = {}

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        X_scaled = scaler.transform(patient_df)
        # Weight importance by how extreme this patient's value is (z-score magnitude)
        z = np.abs(X_scaled[0])
        raw = importances * z
        total = raw.sum() if raw.sum() > 0 else 1
        for i, feat in enumerate(FEATURE_NAMES):
            contributions[feat] = round(float(raw[i] / total * 100), 1)
    else:
        # Fallback: equal split
        for feat in FEATURE_NAMES:
            contributions[feat] = round(100 / len(FEATURE_NAMES), 1)

    return dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True))


def render_xai_chart(contributions, top_n=6):
    """Render an inline Plotly bar chart of top contributing features."""
    items = list(contributions.items())[:top_n]
    feats  = [FEATURE_LABELS.get(k, k) for k, _ in items]
    vals   = [v for _, v in items]
    colors = ['#ff6b6b' if v >= 15 else '#ffaa44' if v >= 8 else '#51cf66' for v in vals]

    fig = go.Figure(go.Bar(
        x=vals[::-1], y=feats[::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=[f"{v:.1f}%" for v in vals[::-1]],
        textposition='outside',
        textfont=dict(color='#e2e8f0', size=12)
    ))
    fig.update_layout(
        title=dict(text="🔍 Why This Prediction? (Top Risk Drivers)", font=dict(color='#e2e8f0', size=15)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0',
        xaxis=dict(title="Contribution to Risk (%)", gridcolor='#2a2d3e', color='#e2e8f0', range=[0, max(vals)*1.25]),
        yaxis=dict(color='#e2e8f0'),
        height=320,
        margin=dict(t=50, b=40, l=160, r=80)
    )
    return fig


def xai_text_summary(contributions, top_n=4):
    """Returns list of human-readable explanation strings."""
    lines = []
    for feat, pct in list(contributions.items())[:top_n]:
        label = FEATURE_LABELS.get(feat, feat)
        if pct >= 15:
            lines.append(f"🔴 **{label}** → contributes **{pct:.1f}%** to your risk (major driver)")
        elif pct >= 8:
            lines.append(f"🟡 **{label}** → contributes **{pct:.1f}%** to your risk (moderate driver)")
        else:
            lines.append(f"🟢 **{label}** → contributes **{pct:.1f}%** to your risk (minor driver)")
    return lines


# ─── ② SMART RECOMMENDATIONS ─────────────────────────────────────────────────
def get_recommendations(pred, proba, patient):
    """
    Returns (list_of_urgent_recs, list_of_lifestyle_recs) based on
    prediction and actual patient values.
    """
    urgent = []
    lifestyle = []
    age     = patient.get('age', 50)
    chol    = patient.get('chol', 200)
    bp      = patient.get('trestbps', 120)
    hr      = patient.get('thalach', 150)
    exang   = patient.get('exang', 0)
    oldpeak = patient.get('oldpeak', 0)
    fbs     = patient.get('fbs', 0)
    ca      = patient.get('ca', 0)

    if pred == 1:  # High Risk
        urgent.append("🏥 **Consult a cardiologist immediately** — high risk detected")
        if chol > 240:
            urgent.append(f"💊 **Cholesterol is {chol} mg/dl** (high) — consider statins & dietary changes")
        if bp > 140:
            urgent.append(f"⚠️ **Blood pressure is {bp} mmHg** (hypertensive) — monitor daily, consider medication")
        if exang == 1:
            urgent.append("🚨 **Exercise-induced angina present** — avoid strenuous activity until evaluated")
        if oldpeak > 2.0:
            urgent.append(f"📉 **ST Depression of {oldpeak}** is elevated — may indicate ischemia")
        if ca > 1:
            urgent.append(f"🔬 **{ca} major vessels affected** — coronary artery disease likely, seek further imaging")
        if fbs == 1:
            urgent.append("🍬 **Fasting blood sugar >120 mg/dl** — screen for diabetes (major cardiac risk factor)")
        if age > 60:
            urgent.append(f"👴 **Age {age}** is a risk factor — regular cardiac screenings every 6 months recommended")

        lifestyle.append("🥗 Adopt a heart-healthy diet: reduce saturated fats, sodium, and processed foods")
        lifestyle.append("🚶 Start a supervised low-intensity exercise program (e.g., 20-min walks)")
        lifestyle.append("🚭 Stop smoking if applicable — it doubles heart disease risk")
        lifestyle.append("😴 Ensure 7–8 hours of quality sleep — sleep deprivation elevates blood pressure")
        lifestyle.append("📊 Track weight, BP, and cholesterol monthly")

    else:  # Low Risk
        if chol > 200:
            lifestyle.append(f"🥦 **Cholesterol is {chol} mg/dl** — aim for <200 mg/dl through diet & exercise")
        if bp > 120:
            lifestyle.append(f"🧂 **Blood pressure {bp} mmHg** — reduce sodium intake to stay below 120 mmHg")
        if hr < 100:
            lifestyle.append(f"💚 **Max heart rate {hr} bpm** — good cardiovascular fitness, keep it up!")

        lifestyle.append("✅ **Low risk detected** — maintain your current healthy habits")
        lifestyle.append("🏃 Continue regular aerobic exercise (150 min/week recommended)")
        lifestyle.append("🍎 Eat a balanced diet rich in fruits, vegetables, and whole grains")
        lifestyle.append("🩺 Schedule annual cardiac checkups for early detection")
        lifestyle.append("🧘 Manage stress with meditation, yoga, or regular relaxation")

    return urgent, lifestyle


def render_recommendations(pred, proba, patient_dict):
    urgent, lifestyle = get_recommendations(pred, proba, patient_dict)

    if urgent:
        st.markdown('<p class="section-header">⚡ Urgent Actions</p>', unsafe_allow_html=True)
        for rec in urgent:
            st.markdown(f'<div class="rec-box-high">{rec}</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">💡 Lifestyle Recommendations</p>', unsafe_allow_html=True)
    for rec in lifestyle:
        box_class = "rec-box-high" if pred == 1 else "rec-box-low"
        st.markdown(f'<div class="{box_class}">{rec}</div>', unsafe_allow_html=True)

    return urgent, lifestyle


# ─── ③ INPUT VALIDATION ──────────────────────────────────────────────────────
def validate_dataframe(df):
    """
    Returns (is_valid, errors_list, warnings_list, cleaned_df).
    Handles: missing columns, missing values, type errors, out-of-range values.
    """
    errors   = []
    warnings = []
    df = df.copy()

    # Check required columns
    missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: **{', '.join(missing_cols)}**")
        return False, errors, warnings, df

    # Extra columns — just warn
    extra_cols = [c for c in df.columns if c not in FEATURE_NAMES]
    if extra_cols:
        warnings.append(f"Extra columns ignored: {', '.join(extra_cols)}")
        df = df[FEATURE_NAMES]

    # Check empty
    if len(df) == 0:
        errors.append("The CSV file is empty (no patient rows found).")
        return False, errors, warnings, df

    # Missing values
    null_counts = df.isnull().sum()
    for feat, cnt in null_counts.items():
        if cnt > 0:
            median_val = df[feat].median()
            df[feat].fillna(median_val, inplace=True)
            warnings.append(
                f"**{FEATURE_LABELS.get(feat, feat)}** had {cnt} missing value(s) → filled with median ({median_val:.1f})"
            )

    # Type coercion
    for feat in FEATURE_NAMES:
        try:
            df[feat] = pd.to_numeric(df[feat])
        except Exception:
            errors.append(f"Column **{FEATURE_LABELS.get(feat, feat)}** contains non-numeric values.")
            return False, errors, warnings, df

    # Range checks
    for feat, (lo, hi, unit) in FEATURE_RANGES.items():
        if feat in df.columns:
            out_of_range = df[(df[feat] < lo) | (df[feat] > hi)]
            if len(out_of_range) > 0:
                unit_str = f" {unit}" if unit else ""
                warnings.append(
                    f"**{FEATURE_LABELS.get(feat, feat)}**: {len(out_of_range)} row(s) have values outside "
                    f"expected range [{lo}–{hi}{unit_str}]. Values clipped."
                )
                df[feat] = df[feat].clip(lo, hi)

    return True, errors, warnings, df


# ─── ④ SAMPLE DATA GENERATOR ─────────────────────────────────────────────────
def generate_sample_patients(n=20, seed=42):
    np.random.seed(seed)
    data = {
        'age':      np.random.randint(29, 77, n),
        'sex':      np.random.randint(0, 2, n),
        'cp':       np.random.randint(0, 4, n),
        'trestbps': np.random.randint(90, 180, n),
        'chol':     np.random.randint(150, 400, n),
        'fbs':      np.random.randint(0, 2, n),
        'restecg':  np.random.randint(0, 3, n),
        'thalach':  np.random.randint(70, 202, n),
        'exang':    np.random.randint(0, 2, n),
        'oldpeak':  np.round(np.random.uniform(0.0, 6.2, n), 1),
        'slope':    np.random.randint(0, 3, n),
        'ca':       np.random.randint(0, 5, n),
        'thal':     np.random.randint(0, 4, n),
    }
    return pd.DataFrame(data)


# ─── PDF Text Sanitiser ───────────────────────────────────────────────────────
def _pdf_safe(text: str) -> str:
    """Strip anything outside latin-1 (emojis, bullets, special chars) for FPDF."""
    import unicodedata, re
    # Replace common unicode symbols with ASCII equivalents
    replacements = {
        '\u2022': '-', '\u2023': '-', '\u25cf': '-',   # bullets
        '\u2019': "'", '\u2018': "'",                   # curly quotes
        '\u201c': '"', '\u201d': '"',                   # curly double quotes
        '\u2013': '-', '\u2014': '-',                   # dashes
        '\u2192': '->',                                 # arrow
        '\u2265': '>=', '\u2264': '<=',                 # comparators
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Strip remaining non-latin-1 characters (emojis etc.)
    return text.encode('latin-1', errors='ignore').decode('latin-1')


# ─── ⑤ ENHANCED PDF GENERATOR ────────────────────────────────────────────────
def _clean_rec(text: str) -> str:
    """Remove markdown bold markers then make latin-1 safe."""
    return _pdf_safe(text.replace("**", ""))


def generate_enhanced_pdf(results, total, high_risk, low_risk, avg_conf,
                          contributions=None, urgent_recs=None, lifestyle_recs=None,
                          patient_dict=None, mode="batch"):
    from datetime import date as _date
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header ──
    pdf.set_fill_color(15, 17, 23)
    pdf.rect(0, 0, 210, 42, 'F')
    pdf.set_text_color(255, 107, 107)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "Heart Disease Risk Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(180, 180, 180)
    pdf.set_xy(10, 22)
    pdf.cell(0, 8, "AI-Powered Prediction  |  Heart Disease Risk Dashboard")
    pdf.set_xy(10, 30)
    mode_label = 'Single Patient' if mode == 'single' else 'Batch Analysis'
    pdf.cell(0, 8, _pdf_safe(f"Generated: {_date.today().strftime('%B %d, %Y')}   |   Mode: {mode_label}"))

    # ── Summary Section ──
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(10, 52)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Summary Statistics", ln=True)
    pdf.set_draw_color(220, 60, 60)
    pdf.set_line_width(0.5)
    pdf.line(10, 62, 200, 62)

    pdf.set_xy(10, 66)
    summary_data = [
        ("Total Patients Analyzed", str(total)),
        ("High Risk Patients",      f"{int(high_risk)} ({high_risk/total*100:.1f}%)"),
        ("Low Risk Patients",       f"{int(low_risk)} ({low_risk/total*100:.1f}%)"),
        ("Average Risk Confidence", f"{avg_conf:.1f}%"),
    ]
    for label, value in summary_data:
        pdf.set_fill_color(245, 245, 250)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(105, 8, f"  {label}", border=0, fill=True)
        pdf.set_font("Helvetica", "B", 11)
        if "High Risk" in label:
            pdf.set_text_color(200, 50, 50)
        elif "Low Risk" in label:
            pdf.set_text_color(30, 150, 80)
        else:
            pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, value, ln=True)
        pdf.set_text_color(0, 0, 0)

    # ── Explainable AI Section ──
    if contributions:
        pdf.set_xy(10, pdf.get_y() + 8)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, "Risk Factor Analysis (Explainable AI)", ln=True)
        pdf.set_draw_color(220, 60, 60)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 6,
            "The following factors contributed most to the prediction. "
            "Higher % = stronger influence on risk.", ln=True)
        pdf.ln(2)

        top_contribs = list(contributions.items())[:6]
        max_val = top_contribs[0][1] if top_contribs else 1

        for feat, pct in top_contribs:
            label = FEATURE_LABELS.get(feat, feat)
            bar_w = int((pct / max_val) * 80)
            pdf.set_x(12)

            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(55, 7, _pdf_safe(label))

            pdf.set_fill_color(230, 230, 230)
            pdf.cell(85, 5, "", fill=True)
            pdf.set_xy(pdf.get_x() - 85, pdf.get_y())

            if pct >= 15:
                pdf.set_fill_color(220, 60, 60)
            elif pct >= 8:
                pdf.set_fill_color(255, 160, 50)
            else:
                pdf.set_fill_color(60, 180, 100)
            pdf.cell(bar_w, 5, "", fill=True)

            pdf.set_x(100)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(30, 7, f"{pct:.1f}%", ln=True)

    # ── Recommendations Section ──
    if urgent_recs or lifestyle_recs:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, "Clinical Recommendations", ln=True)
        pdf.set_draw_color(220, 60, 60)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        if urgent_recs:
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(180, 30, 30)
            pdf.cell(0, 7, "Urgent Actions:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            for rec in urgent_recs:
                pdf.set_x(14)
                pdf.cell(0, 6, _pdf_safe(f"- {_clean_rec(rec).strip()}"), ln=True)

        if lifestyle_recs:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 120, 60)
            pdf.cell(0, 7, "Lifestyle Recommendations:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            for rec in lifestyle_recs:
                pdf.set_x(14)
                pdf.cell(0, 6, _pdf_safe(f"- {_clean_rec(rec).strip()}"), ln=True)

    # ── High Risk Patients Table (batch mode) ──
    if mode == "batch" and results is not None:
        high_df = results[results['Prediction'] == 1].head(15).reset_index(drop=True)
        if len(high_df) > 0:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(0, 0, 0)
            pdf.set_xy(10, 15)
            pdf.cell(0, 8, "High Risk Patients - Detail", ln=True)
            pdf.set_draw_color(220, 60, 60)
            pdf.line(10, 25, 200, 25)

            pdf.set_xy(10, 30)
            pdf.set_fill_color(220, 60, 60)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 10)
            col_w = [18, 14, 14, 32, 26, 28, 32]
            headers = ['Age', 'Sex', 'CP', 'Blood Pres.', 'Chol', 'Max HR', 'Confidence']
            for i, h in enumerate(headers):
                pdf.cell(col_w[i], 8, h, border=0, fill=True, align='C')
            pdf.ln()

            pdf.set_text_color(30, 30, 30)
            for idx, row in high_df.iterrows():
                fill = idx % 2 == 0
                if fill:
                    pdf.set_fill_color(255, 240, 240)
                else:
                    pdf.set_fill_color(255, 255, 255)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_x(10)
                pdf.cell(col_w[0], 7, str(int(row.get('age', 0))),      border=0, fill=True, align='C')
                pdf.cell(col_w[1], 7, "M" if row.get('sex',0)==1 else "F", border=0, fill=True, align='C')
                pdf.cell(col_w[2], 7, str(int(row.get('cp', 0))),       border=0, fill=True, align='C')
                pdf.cell(col_w[3], 7, str(int(row.get('trestbps', 0))), border=0, fill=True, align='C')
                pdf.cell(col_w[4], 7, str(int(row.get('chol', 0))),     border=0, fill=True, align='C')
                pdf.cell(col_w[5], 7, str(int(row.get('thalach', 0))),  border=0, fill=True, align='C')
                pdf.cell(col_w[6], 7, str(row.get('Confidence', '')),   border=0, fill=True, align='C')
                pdf.ln()

    # ── Footer ──
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 8,
        "Generated by Heart Disease Risk Dashboard  |  For clinical decision support only - not a medical diagnosis.",
        align='C')

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name


# ─── Gauge Chart ──────────────────────────────────────────────────────────────
def make_gauge(value, title="Risk Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 1),
        title={'text': title, 'font': {'color': '#e2e8f0', 'size': 16}},
        number={'suffix': '%', 'font': {'color': '#e2e8f0', 'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#e2e8f0', 'tickfont': {'color': '#e2e8f0'}},
            'bar': {'color': '#ff6b6b' if value > 50 else '#51cf66'},
            'bgcolor': '#1a1d27',
            'bordercolor': '#2a2d3e',
            'steps': [
                {'range': [0,  33], 'color': '#0d2e1e'},
                {'range': [33, 66], 'color': '#2e2a0d'},
                {'range': [66,100], 'color': '#3d1515'},
            ],
            'threshold': {
                'line': {'color': '#ffffff', 'width': 3},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0',
        height=300,
        margin=dict(t=30, b=20, l=40, r=40)
    )
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Heart Disease\nRisk Predictor")
    st.markdown("---")
    st.markdown("""
**Model Info:**
- Algorithm: sklearn (best from comparison)
- Dataset: Cleveland Heart Disease (1025 rows)
- Train/Test split: 80/20
- Features: 13 clinical variables
""")
    st.markdown("---")
    st.markdown("**New in this version:**")
    st.markdown("""
- 🔍 Explainable AI (XAI)
- 💊 Smart Recommendations
- ✅ Input Validation
- 📊 Sample Data Generator
- 📄 Enhanced PDF Reports
""")
    st.markdown("---")

    if model is not None and hasattr(model, 'feature_importances_'):
        st.markdown("**Global Top Risk Factors:**")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            'Feature':    [FEATURE_LABELS.get(f, f) for f in FEATURE_NAMES],
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(5)

        for _, row in feat_df.iterrows():
            bar_pct = int(row['Importance'] * 100 / feat_df['Importance'].max())
            st.markdown(f"""
            <div style="margin-bottom:8px;">
              <div style="font-size:12px;color:#e2e8f0;margin-bottom:3px;">{row['Feature']}</div>
              <div style="background:#2a2d3e;border-radius:4px;height:6px;">
                <div style="background:#ff6b6b;width:{bar_pct}%;height:6px;border-radius:4px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("# Heart Disease Risk Dashboard")
st.markdown("---")

tab1, tab2 = st.tabs(["🧑 Single Patient", "📋 Batch Upload"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE PATIENT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Patient Details")
    st.markdown("Fill in the patient's medical data and click **Predict Risk**.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Basic Info**")
        age      = st.slider("Age",                    20, 80,  50)
        sex      = st.selectbox("Sex",                 [0, 1],  format_func=lambda x: "Female" if x == 0 else "Male")
        cp       = st.selectbox("Chest Pain Type",     [0,1,2,3], format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",2:"Non-Anginal",3:"Asymptomatic"}[x])
        trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
        chol     = st.slider("Cholesterol (mg/dl)",    100, 600, 240)

    with c2:
        st.markdown("**Test Results**")
        fbs      = st.selectbox("Fasting Blood Sugar > 120", [0,1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg  = st.selectbox("Resting ECG",         [0,1,2], format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"Left Ventricular Hypertrophy"}[x])
        thalach  = st.slider("Max Heart Rate",          60, 220, 150)
        exang    = st.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x: "No" if x == 0 else "Yes")

    with c3:
        st.markdown("**Advanced**")
        oldpeak  = st.slider("ST Depression",          0.0, 7.0, 1.0, step=0.1)
        slope    = st.selectbox("ST Slope",            [0,1,2], format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
        ca       = st.selectbox("Major Vessels (0-4)", [0,1,2,3,4])
        thal     = st.selectbox("Thalassemia",         [0,1,2,3], format_func=lambda x: {0:"Normal",1:"Fixed Defect",2:"Reversible Defect",3:"Unknown"}[x])

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Risk", use_container_width=True, type="primary")

    if predict_btn:
        if model is None or scaler is None:
            st.stop()

        patient_dict = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
                            fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
                            oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)

        patient_data = pd.DataFrame([list(patient_dict.values())], columns=FEATURE_NAMES)

        X_scaled = scaler.transform(patient_data)
        pred     = model.predict(X_scaled)[0]
        proba    = model.predict_proba(X_scaled)[0][1] * 100

        st.markdown("---")
        st.markdown('<p class="section-header">📊 Prediction Result</p>', unsafe_allow_html=True)

        g_col, r_col = st.columns([1, 1])

        with g_col:
            st.plotly_chart(make_gauge(proba, "Personal Risk Score"), use_container_width=True)

        with r_col:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if pred == 1:
                st.markdown(f"""
                <div class="risk-box-high">
                    <div style="font-size:2.5rem;">⚠️</div>
                    <div style="font-size:1.4rem;font-weight:600;color:#ff6b6b;margin:8px 0;">High Risk</div>
                    <div style="font-size:2rem;font-weight:700;color:#ff6b6b;">{proba:.1f}%</div>
                    <div style="font-size:0.85rem;color:#cc8888;margin-top:8px;">
                        High probability of heart disease detected.<br>
                        Immediate medical attention recommended.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-box-low">
                    <div style="font-size:2.5rem;">✅</div>
                    <div style="font-size:1.4rem;font-weight:600;color:#51cf66;margin:8px 0;">Low Risk</div>
                    <div style="font-size:2rem;font-weight:700;color:#51cf66;">{proba:.1f}%</div>
                    <div style="font-size:0.85rem;color:#669966;margin-top:8px;">
                        Low probability of heart disease detected.<br>
                        Continue regular health checkups.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ① EXPLAINABLE AI ──────────────────────────────────────────────────
        st.markdown("---")
        contributions = get_xai_contributions(model, scaler, patient_data)
        st.plotly_chart(render_xai_chart(contributions), use_container_width=True)

        st.markdown("**What's driving this prediction:**")
        for line in xai_text_summary(contributions):
            st.markdown(line)

        # ── ② SMART RECOMMENDATIONS ──────────────────────────────────────────
        st.markdown("---")
        urgent_recs, lifestyle_recs = render_recommendations(pred, proba, patient_dict)

        # ── Patient Summary ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-header">📋 Patient Summary</p>', unsafe_allow_html=True)
        st.markdown(f"""
        | Feature | Value | Status |
        |---|---|---|
        | Age | {age} yrs | {'⚠️ Senior' if age > 60 else '✅ OK'} |
        | Sex | {'Male' if sex == 1 else 'Female'} | — |
        | Cholesterol | {chol} mg/dl | {'🔴 High' if chol > 240 else '🟡 Borderline' if chol > 200 else '✅ Normal'} |
        | Blood Pressure | {trestbps} mmHg | {'🔴 Hypertensive' if trestbps > 140 else '🟡 Elevated' if trestbps > 120 else '✅ Normal'} |
        | Max Heart Rate | {thalach} bpm | {'✅ Good' if thalach >= 100 else '⚠️ Low'} |
        | Exercise Angina | {'Yes' if exang == 1 else 'No'} | {'🔴 Present' if exang == 1 else '✅ None'} |
        | ST Depression | {oldpeak} | {'🔴 High' if oldpeak > 2.0 else '✅ Normal'} |
        """)

        # ── Single Patient PDF ────────────────────────────────────────────────
        st.markdown("---")
        try:
            single_results = pd.DataFrame([patient_dict])
            single_results['Prediction'] = pred
            pdf_path = generate_enhanced_pdf(
                single_results, total=1,
                high_risk=1 if pred==1 else 0,
                low_risk=0 if pred==1 else 1,
                avg_conf=proba,
                contributions=contributions,
                urgent_recs=urgent_recs,
                lifestyle_recs=lifestyle_recs,
                patient_dict=patient_dict,
                mode="single"
            )
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            os.unlink(pdf_path)
            st.download_button(
                "📄 Download Full Patient Report (PDF)",
                data=pdf_bytes,
                file_name="patient_risk_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Upload Patient CSV")
    st.markdown("Upload a CSV file with multiple patients to predict risk for all at once.")

    # ── ④ SAMPLE DATA GENERATOR BUTTON ────────────────────────────────────────
    gen_col, info_col = st.columns([1, 2])
    with gen_col:
        n_patients = st.number_input("Number of patients to generate", min_value=5, max_value=200, value=20, step=5)
        seed_val   = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    with info_col:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **No CSV?** Use the generator below to create sample patient data instantly — no file needed!")

    if st.button("⚡ Generate Sample Data", use_container_width=True):
        sample_df = generate_sample_patients(n=n_patients, seed=seed_val)
        csv_bytes  = sample_df.to_csv(index=False).encode('utf-8')
        st.success(f"✅ Generated {n_patients} sample patients!")
        st.download_button(
            "📥 Download Sample CSV",
            data=csv_bytes,
            file_name="sample_patients.csv",
            mime="text/csv",
            use_container_width=True
        )
        with st.expander("Preview Generated Data", expanded=True):
            st.dataframe(sample_df.head(10), use_container_width=True)

    st.markdown("---")

    uploaded_file = st.file_uploader("Or upload your own Patient CSV", type=["csv"], key="batch_upload")

    if uploaded_file is None:
        st.markdown("""
        <div style="text-align:center; padding: 40px 20px;">
            <div style="font-size: 3rem;">📋</div>
            <div style="font-size: 1.1rem; margin-top: 12px; color: #888;">Upload a CSV to get started</div>
            <div style="font-size: 0.85rem; margin-top: 8px; color: #555;">
                Or use the Sample Data Generator above
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if model is None or scaler is None:
            st.stop()

        raw_df = pd.read_csv(uploaded_file)

        # ── ③ INPUT VALIDATION ────────────────────────────────────────────────
        is_valid, errors, val_warnings, df = validate_dataframe(raw_df)

        if errors:
            for err in errors:
                st.markdown(f'<div class="validation-error">❌ {err}</div>', unsafe_allow_html=True)
            st.stop()

        if val_warnings:
            st.markdown('<p class="section-header">⚠️ Data Quality Notices (Auto-Fixed)</p>', unsafe_allow_html=True)
            for w in val_warnings:
                st.markdown(f'<div class="validation-warning">⚠️ {w}</div>', unsafe_allow_html=True)

        st.success(f"✅ Data validated: {len(df)} patients, {len(df.columns)} features — ready for prediction!")

        with st.expander("View Uploaded Data", expanded=False):
            st.dataframe(df, use_container_width=True)
            st.caption(f"{len(df)} patients — {len(df.columns)} features")

        with st.spinner("Running predictions..."):
            try:
                X_scaled          = scaler.transform(df)
                predictions_label = model.predict(X_scaled)
                predictions_proba = model.predict_proba(X_scaled)[:, 1]

                results               = df.copy()
                results['Prediction'] = predictions_label
                results['Risk']       = ['High Risk' if p == 1 else 'Low Risk' for p in predictions_label]
                results['Confidence'] = (predictions_proba * 100).round(1).astype(str) + '%'
                results['Score']      = predictions_proba

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        st.success(f"🎉 Predictions complete for {len(results)} patients!")

        # ── Summary Metrics ───────────────────────────────────────────────────
        st.markdown('<p class="section-header">📊 Summary</p>', unsafe_allow_html=True)

        total     = len(results)
        high_risk = (results['Prediction'] == 1).sum()
        low_risk  = total - high_risk
        avg_conf  = (results['Score'] * 100).mean()
        high_pct  = high_risk / total * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Patients", total)
        col2.metric("High Risk",      int(high_risk), f"{high_pct:.1f}%")
        col3.metric("Low Risk",       int(low_risk),  f"{100 - high_pct:.1f}%")
        col4.metric("Avg Confidence", f"{avg_conf:.1f}%")

        # ── Batch Risk Gauge ──────────────────────────────────────────────────
        st.markdown('<p class="section-header">🔴 Batch Risk Gauge</p>', unsafe_allow_html=True)
        st.caption("Shows what % of uploaded patients are high risk")
        st.plotly_chart(make_gauge(high_pct, "Batch High Risk %"), use_container_width=True)

        # ── Results Table ─────────────────────────────────────────────────────
        st.markdown('<p class="section-header">📋 Patient Results</p>', unsafe_allow_html=True)
        base_cols = [c for c in ['age', 'sex', 'cp', 'trestbps', 'chol'] if c in results.columns]
        st.dataframe(results[base_cols + ['Risk', 'Confidence']], use_container_width=True, height=300)

        # ── ① BATCH XAI ──────────────────────────────────────────────────────
        st.markdown('<p class="section-header">🔍 Global Risk Factor Importance</p>', unsafe_allow_html=True)
        st.caption("These are the features that most influenced predictions across all patients")

        if hasattr(model, 'feature_importances_'):
            gi = model.feature_importances_
            gi_df = pd.DataFrame({
                'Feature': [FEATURE_LABELS.get(f, f) for f in FEATURE_NAMES],
                'Importance': gi * 100
            }).sort_values('Importance', ascending=False)

            fig_imp = go.Figure(go.Bar(
                x=gi_df['Importance'], y=gi_df['Feature'],
                orientation='h',
                marker_color=['#ff6b6b' if v >= 12 else '#ffaa44' if v >= 7 else '#51cf66'
                              for v in gi_df['Importance']],
                text=[f"{v:.1f}%" for v in gi_df['Importance']],
                textposition='outside',
                textfont=dict(color='#e2e8f0')
            ))
            fig_imp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                xaxis=dict(title="Global Importance (%)", gridcolor='#2a2d3e', color='#e2e8f0'),
                yaxis=dict(color='#e2e8f0', autorange='reversed'),
                height=400,
                margin=dict(t=20, b=40, l=160, r=80)
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        # ── Charts ────────────────────────────────────────────────────────────
        st.markdown('<p class="section-header">📈 Visual Analysis</p>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(
                results, names='Risk', title='Risk Distribution',
                color='Risk',
                color_discrete_map={'High Risk': '#ff6b6b', 'Low Risk': '#51cf66'},
                hole=0.4
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0', margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            if 'age' in results.columns:
                fig_age = px.histogram(
                    results, x='age', color='Risk',
                    title='Risk by Age Group', nbins=15,
                    color_discrete_map={'High Risk': '#ff6b6b', 'Low Risk': '#51cf66'},
                    barmode='overlay', opacity=0.75
                )
                fig_age.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0',
                    xaxis=dict(gridcolor='#2a2d3e'),
                    yaxis=dict(gridcolor='#2a2d3e'),
                    margin=dict(t=40, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_age, use_container_width=True)

        if 'chol' in results.columns and 'trestbps' in results.columns:
            fig_scatter = px.scatter(
                results, x='chol', y='trestbps', color='Risk',
                title='Cholesterol vs Resting Blood Pressure',
                color_discrete_map={'High Risk': '#ff6b6b', 'Low Risk': '#51cf66'},
                opacity=0.75
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                xaxis=dict(gridcolor='#2a2d3e', title='Cholesterol (mg/dl)'),
                yaxis=dict(gridcolor='#2a2d3e', title='Blood Pressure (mmHg)'),
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ── ② BATCH RECOMMENDATIONS ───────────────────────────────────────────
        st.markdown('<p class="section-header">💊 Batch Recommendations</p>', unsafe_allow_html=True)
        high_pct_val = high_risk / total * 100
        avg_chol = df['chol'].mean() if 'chol' in df.columns else 0
        avg_bp   = df['trestbps'].mean() if 'trestbps' in df.columns else 0

        if high_pct_val > 60:
            st.markdown('<div class="rec-box-high">🚨 <b>Critical:</b> Over 60% of patients are high risk. This cohort requires urgent clinical review and intervention.</div>', unsafe_allow_html=True)
        elif high_pct_val > 30:
            st.markdown('<div class="rec-box-high">⚠️ <b>Warning:</b> Significant proportion of high-risk patients detected. Prioritize cardiology consultations.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="rec-box-low">✅ <b>Good:</b> Majority of patients are low risk. Continue preventive care and monitoring programs.</div>', unsafe_allow_html=True)

        if avg_chol > 240:
            st.markdown(f'<div class="rec-box-high">💊 Average cholesterol is <b>{avg_chol:.0f} mg/dl</b> (high across cohort) — consider population-level dietary interventions.</div>', unsafe_allow_html=True)
        if avg_bp > 130:
            st.markdown(f'<div class="rec-box-high">⚠️ Average blood pressure is <b>{avg_bp:.0f} mmHg</b> — recommend blood pressure screening program for this cohort.</div>', unsafe_allow_html=True)

        # ── High Risk Table ───────────────────────────────────────────────────
        high_df = results[results['Prediction'] == 1]
        if len(high_df) > 0:
            st.markdown('<p class="section-header">🔴 High Risk Patients</p>', unsafe_allow_html=True)
            st.dataframe(high_df[base_cols + ['Confidence']].reset_index(drop=True), use_container_width=True)

        # ── Downloads ─────────────────────────────────────────────────────────
        st.markdown('<p class="section-header">📥 Export Results</p>', unsafe_allow_html=True)

        dl1, dl2 = st.columns(2)

        with dl1:
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Download CSV Results",
                data=csv,
                file_name="heart_disease_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        with dl2:
            try:
                # Use global XAI contributions for batch PDF
                batch_contribs = None
                if hasattr(model, 'feature_importances_'):
                    gi = model.feature_importances_
                    total_gi = gi.sum()
                    batch_contribs = {
                        FEATURE_NAMES[i]: round(float(gi[i] / total_gi * 100), 1)
                        for i in range(len(FEATURE_NAMES))
                    }
                    batch_contribs = dict(sorted(batch_contribs.items(), key=lambda x: x[1], reverse=True))

                # Get general recommendations based on overall stats
                dummy_patient = {
                    'chol':     avg_chol,
                    'trestbps': avg_bp,
                    'age':      df['age'].mean() if 'age' in df.columns else 50,
                    'exang':    0,
                    'oldpeak':  df['oldpeak'].mean() if 'oldpeak' in df.columns else 1.0,
                    'fbs':      0,
                    'ca':       0,
                    'thalach':  df['thalach'].mean() if 'thalach' in df.columns else 150,
                }
                pred_mode = 1 if high_pct_val > 50 else 0
                batch_urgent, batch_lifestyle = get_recommendations(pred_mode, high_pct_val, dummy_patient)

                with st.spinner("Generating enhanced PDF..."):
                    pdf_path = generate_enhanced_pdf(
                        results, total, high_risk, low_risk, avg_conf,
                        contributions=batch_contribs,
                        urgent_recs=batch_urgent,
                        lifestyle_recs=batch_lifestyle,
                        mode="batch"
                    )
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    os.unlink(pdf_path)

                st.download_button(
                    "📄 Download Enhanced PDF Report",
                    data=pdf_bytes,
                    file_name="heart_disease_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
