import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import tempfile, os, warnings
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

# ─── Feature Labels ───────────────────────────────────────────────────────────
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

# ─── PDF Generator ────────────────────────────────────────────────────────────
def generate_pdf(results, total, high_risk, low_risk, avg_conf):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_fill_color(15, 17, 23)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(10, 10)
    pdf.cell(0, 10, "Heart Disease Risk Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 24)
    pdf.cell(0, 8, "AI Application Lab Project - Automated Prediction Report")

    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(10, 50)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Summary", ln=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, 60, 200, 60)

    pdf.set_xy(10, 64)
    summary_data = [
        ("Total Patients Analyzed", str(total)),
        ("High Risk Patients",      f"{int(high_risk)} ({high_risk/total*100:.1f}%)"),
        ("Low Risk Patients",       f"{int(low_risk)} ({low_risk/total*100:.1f}%)"),
        ("Average Confidence",      f"{avg_conf:.1f}%"),
    ]
    for label, value in summary_data:
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(100, 8, label)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, value, ln=True)

    pdf.set_xy(10, 110)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "High Risk Patients", ln=True)
    pdf.line(10, 120, 200, 120)

    high_df = results[results['Prediction'] == 1].reset_index(drop=True)

    if len(high_df) == 0:
        pdf.set_xy(10, 124)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, "No high risk patients found.")
    else:
        pdf.set_xy(10, 124)
        pdf.set_fill_color(220, 220, 220)
        pdf.set_font("Helvetica", "B", 10)
        col_w = [20, 15, 15, 35, 30, 35]
        headers = ['Age', 'Sex', 'CP', 'Blood Pres.', 'Chol', 'Confidence']
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 8, h, border=1, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 10)
        for _, row in high_df.head(15).iterrows():
            pdf.set_x(10)
            pdf.cell(col_w[0], 7, str(int(row.get('age', 0))),      border=1)
            pdf.cell(col_w[1], 7, str(int(row.get('sex', 0))),      border=1)
            pdf.cell(col_w[2], 7, str(int(row.get('cp', 0))),       border=1)
            pdf.cell(col_w[3], 7, str(int(row.get('trestbps', 0))), border=1)
            pdf.cell(col_w[4], 7, str(int(row.get('chol', 0))),     border=1)
            pdf.cell(col_w[5], 7, str(row.get('Confidence', '')),   border=1)
            pdf.ln()

    pdf.set_xy(10, 275)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 8, "Generated by Heart Disease Risk Dashboard - AI Application Lab")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Heart Disease\nRisk Predictor")
    st.markdown("---")
    st.markdown("""
**Model Info:**
- Algorithm: sklearn (best from comparison)
- Dataset: Cleveland Heart Disease (1025 rows)
- Train/Test split: 80/20
""")
    st.markdown("---")

    # Feature importance in sidebar
    if model is not None and hasattr(model, 'feature_importances_'):
        st.markdown("**Top Risk Factors:**")
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

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Single Patient", "Batch Upload"])

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
    predict_btn = st.button("Predict Risk", use_container_width=True, type="primary")

    if predict_btn:
        if model is None or scaler is None:
            st.stop()

        patient_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                                       restecg, thalach, exang, oldpeak,
                                       slope, ca, thal]], columns=FEATURE_NAMES)

        X_scaled = scaler.transform(patient_data)
        pred     = model.predict(X_scaled)[0]
        proba    = model.predict_proba(X_scaled)[0][1] * 100

        st.markdown("---")
        st.markdown('<p class="section-header">Prediction Result</p>', unsafe_allow_html=True)

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
                        This patient has a high probability of heart disease.<br>
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
                        This patient has a low probability of heart disease.<br>
                        Continue regular health checkups.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Key inputs summary
            st.markdown("**Patient Summary:**")
            st.markdown(f"""
            | Feature | Value |
            |---|---|
            | Age | {age} |
            | Sex | {'Male' if sex == 1 else 'Female'} |
            | Cholesterol | {chol} mg/dl |
            | Blood Pressure | {trestbps} mmHg |
            | Max Heart Rate | {thalach} bpm |
            """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Upload Patient CSV")
    st.markdown("Upload a CSV file with multiple patients to predict risk for all at once.")

    uploaded_file = st.file_uploader("Upload Patient CSV", type=["csv"], key="batch_upload")

    if uploaded_file is None:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px;">
            <div style="font-size: 3rem;">📋</div>
            <div style="font-size: 1.1rem; margin-top: 12px; color: #888;">Upload a CSV to get started</div>
            <div style="font-size: 0.85rem; margin-top: 8px; color: #555;">
                Run generate_sample.py to create test data
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if model is None or scaler is None:
            st.stop()

        df = pd.read_csv(uploaded_file)

        with st.expander("View Uploaded Data", expanded=False):
            st.dataframe(df, use_container_width=True)
            st.caption(f"{len(df)} patients - {len(df.columns)} features")

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

        st.success(f"Predictions complete for {len(results)} patients!")

        # ── Summary Metrics ──────────────────────────────────────────────────
        st.markdown('<p class="section-header">Summary</p>', unsafe_allow_html=True)

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

        # ── Batch Risk Gauge ─────────────────────────────────────────────────
        st.markdown('<p class="section-header">Batch Risk Gauge</p>', unsafe_allow_html=True)
        st.caption("Shows what % of your uploaded patients are high risk")
        st.plotly_chart(make_gauge(high_pct, "Batch High Risk %"), use_container_width=True)

        # ── Results Table ────────────────────────────────────────────────────
        st.markdown('<p class="section-header">Patient Results</p>', unsafe_allow_html=True)
        base_cols = [c for c in ['age', 'sex', 'cp', 'trestbps', 'chol'] if c in results.columns]
        st.dataframe(results[base_cols + ['Risk', 'Confidence']], use_container_width=True, height=300)

        # ── Charts ───────────────────────────────────────────────────────────
        st.markdown('<p class="section-header">Visual Analysis</p>', unsafe_allow_html=True)

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

        # ── High Risk Table ──────────────────────────────────────────────────
        high_df = results[results['Prediction'] == 1]
        if len(high_df) > 0:
            st.markdown('<p class="section-header">High Risk Patients</p>', unsafe_allow_html=True)
            st.dataframe(high_df[base_cols + ['Confidence']].reset_index(drop=True), use_container_width=True)

        # ── Downloads ────────────────────────────────────────────────────────
        st.markdown('<p class="section-header">Export Results</p>', unsafe_allow_html=True)

        dl1, dl2 = st.columns(2)

        with dl1:
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="heart_disease_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        with dl2:
            try:
                with st.spinner("Generating PDF..."):
                    pdf_path = generate_pdf(results, total, high_risk, low_risk, avg_conf)
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    os.unlink(pdf_path)

                st.download_button(
                    "Download PDF Report",
                    data=pdf_bytes,
                    file_name="heart_disease_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
