# 🫀 Heart Disease Risk Prediction Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-2ECC71?style=for-the-badge)

**An AI-powered clinical decision support tool for heart disease risk prediction**

[🌐 Live App](https://health-disease-ai.streamlit.app) · [📂 GitHub](https://github.com/divyansh2453/Health_Disease_Dashboard) · [📊 Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

</div>

---

## 📌 Overview

This project is a full-stack AI web application that predicts heart disease risk from clinical patient data using machine learning. Built as part of the **AI Application Lab** course, it goes beyond a basic ML model to deliver a complete, deployable clinical decision support tool.

The system covers the entire ML pipeline — from data exploration and multi-model comparison in Google Colab, to a production-ready Streamlit dashboard with explainable AI, smart recommendations, input validation, and PDF report generation.

---

## ✨ Features

### 🧍 Single Patient Mode
| Feature | Description |
|---|---|
| Input Form | Sliders and dropdowns for all 13 clinical features with human-readable labels |
| Risk Gauge | Personal risk speedometer (0–100%) that turns red or green based on score |
| Result Card | Color-coded High Risk / Low Risk result with confidence score |
| Explainable AI | Bar chart showing which features drove the prediction and by how much |
| XAI Text Summary | Human-readable explanation: "Chest Pain Type contributes 24.3% to your risk" |
| Patient Summary | Table with status indicators — Normal / Elevated / Hypertensive / High |
| Smart Recommendations | Personalized urgent actions and lifestyle advice based on actual patient values |
| PDF Report | Download a full patient report with gauge, XAI chart, and recommendations |

### 📋 Batch Upload Mode
| Feature | Description |
|---|---|
| CSV Upload | Upload a file with any number of patients for bulk prediction |
| Input Validation | Auto-detects missing columns, fills missing values, clips out-of-range data |
| Summary Metrics | Total patients, high risk count, low risk count, average confidence |
| Batch Risk Gauge | Speedometer showing overall % of high-risk patients in the uploaded group |
| Feature Importance | Global bar chart showing which features matter most across all predictions |
| Visual Analysis | Donut chart, age histogram, cholesterol vs BP scatter, confidence distribution |
| Batch Recommendations | Cohort-level insights based on average cholesterol and blood pressure |
| High Risk Table | Filtered table showing only patients requiring urgent attention |
| Export | Download predictions as CSV or enhanced PDF report |
| Sample Generator | Generate realistic random patient data directly in the app — no file needed |

### 📊 Always Visible (Sidebar)
- Top 5 global risk factors as animated progress bars
- Model info and version details
- Feature list of new capabilities

---

## 🔍 Explainable AI (XAI)

One of the standout features of this dashboard is the **per-patient explainability** system. Instead of just returning a risk score, the model explains *why* it made that prediction.

**How it works:**
1. Uses `feature_importances_` from the Gradient Boosting model
2. Weights each feature's global importance by the patient's z-score (how extreme their value is)
3. Normalizes contributions to sum to 100%
4. Renders a color-coded bar chart: 🔴 major driver / 🟡 moderate / 🟢 minor

```
Example output for a 67-year-old patient:
  Chest Pain Type      → 24.3%  🔴 Major driver
  Major Vessels        → 18.7%  🔴 Major driver
  ST Depression        → 14.2%  🔴 Major driver
  Max Heart Rate       →  9.1%  🟡 Moderate driver
  Thalassemia          →  8.3%  🟡 Moderate driver
```

---

## 💊 Smart Recommendations

The system generates **personalized clinical recommendations** based on the prediction and actual patient values — not generic advice.

**High Risk patients receive:**
- Urgent referral triggers (e.g., "Cholesterol is 312 mg/dl — consider statins")
- Exercise restrictions if angina is present
- Diabetes screening if fasting blood sugar is elevated
- Age-specific monitoring frequency

**Low Risk patients receive:**
- Positive reinforcement of healthy metrics
- Preventive care reminders
- Lifestyle optimization tips

---

## ✅ Input Validation

The batch upload system includes a full validation pipeline:

| Check | Action |
|---|---|
| Missing required columns | Show error, block prediction |
| Extra unknown columns | Warn and ignore them |
| Empty CSV | Show error, block prediction |
| Missing values | Auto-fill with column median, warn user |
| Non-numeric values | Show error with column name |
| Out-of-range values | Auto-clip to valid range, warn user |

---

## 🧠 Model

| Property | Value |
|---|---|
| Dataset | Cleveland Heart Disease (1,025 patients) |
| Algorithm | Gradient Boosting Classifier |
| Accuracy | ~85% |
| AUC Score | ~92% |
| F1 Score | ~86% |
| Train/Test Split | 80% / 20% (stratified) |
| Preprocessing | StandardScaler normalization |
| Serialization | pickle protocol 4 |

### Models Compared During Training

| Model | AUC | Rank |
|---|---|---|
| **Gradient Boosting** | **~92%** | **🥇 1st — Selected** |
| Random Forest | ~91% | 🥈 2nd |
| Logistic Regression | ~90% | 🥉 3rd |
| SVM | ~90% | 4th |
| KNN | ~85% | 5th |
| Decision Tree | ~76% | 6th |

### Top 5 Most Important Features
1. Chest Pain Type (cp)
2. Number of Major Vessels (ca)
3. Thalassemia (thal)
4. ST Depression (oldpeak)
5. Max Heart Rate (thalach)

---

## 🗂️ Project Structure

```
Health_Disease_Dashboard/
│
├── app.py                  ← Main Streamlit dashboard (1078 lines)
├── requirements.txt        ← Python dependencies
├── generate_sample.py      ← CLI script to generate test data
├── .python-version         ← Python 3.11
├── model/
│   ├── heart_model.pkl     ← Trained Gradient Boosting model
│   └── scaler.pkl          ← Fitted StandardScaler
└── README.md
```

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/divyansh2453/Health_Disease_Dashboard.git
cd Health_Disease_Dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser 🎉

---

## 📦 Requirements

```
streamlit
pandas
scikit-learn
plotly
fpdf2
numpy
```

---

## 📁 Dataset

**Cleveland Heart Disease Dataset** — UCI Machine Learning Repository

- 📥 [Download on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- 1,025 patients · 14 features · Binary classification
- Class balance: 51.3% positive · 48.7% negative · 0 missing values

### Feature Reference

| Feature | Description | Type |
|---|---|---|
| age | Patient age in years | Numerical |
| sex | Sex (1 = Male, 0 = Female) | Categorical |
| cp | Chest pain type (0–3) | Categorical |
| trestbps | Resting blood pressure (mmHg) | Numerical |
| chol | Serum cholesterol (mg/dl) | Numerical |
| fbs | Fasting blood sugar > 120 mg/dl | Categorical |
| restecg | Resting ECG results (0–2) | Categorical |
| thalach | Maximum heart rate achieved | Numerical |
| exang | Exercise induced angina | Categorical |
| oldpeak | ST depression induced by exercise | Numerical |
| slope | Slope of peak exercise ST segment | Categorical |
| ca | Number of major vessels (0–4) | Numerical |
| thal | Thalassemia type (0–3) | Categorical |
| **target** | **Heart disease present (1=Yes, 0=No)** | **Target** |

---

## 🛠️ Tech Stack

| Category | Tool | Purpose |
|---|---|---|
| Language | Python 3.11 | Core language |
| ML | scikit-learn | Model training, evaluation, preprocessing |
| Web App | Streamlit | Dashboard framework |
| Charts | Plotly | Interactive visualizations and gauge |
| PDF | fpdf2 | Enhanced PDF report generation |
| Data | pandas / numpy | Data processing and analysis |
| Training | Google Colab | Cloud-based model training (free GPU) |
| Storage | Google Drive | Persistent model and dataset storage |

---

## 🧪 Generate Sample Data

**Option 1 — Inside the app (easiest):**
Go to the Batch Upload tab → set number of patients → click "Generate Sample Data"

**Option 2 — Command line:**
```bash
# Default 20 patients
python generate_sample.py

# Custom count
python generate_sample.py --n 50

# Fixed seed for reproducibility
python generate_sample.py --n 20 --seed 42

# Custom output filename
python generate_sample.py --n 100 --output my_patients.csv
```

---

## 🏗️ Development Journey

This project was built iteratively, solving real challenges along the way:

| Challenge | Solution |
|---|---|
| PyCaret incompatible with Python 3.12 on Colab | Switched to scikit-learn (pre-installed, fully compatible) |
| Model files corrupted during Google Drive download | Re-saved with pickle protocol=4 for cross-platform compatibility |
| FPDF Unicode encoding errors with emojis/dashes | Built `_pdf_safe()` sanitizer to strip non-latin-1 characters |
| Streamlit not directly accessible from Colab | Deployed locally via VS Code with Python 3.11, then to Streamlit Cloud |
| Manual CSV upload required each Colab session | Mounted Google Drive with a `BASE` path variable for persistence |

---

## 🔮 Future Enhancements

- [ ] Deploy SHAP values for more precise per-feature explanations
- [ ] Add patient history tracking with a SQLite database
- [ ] Implement user authentication for multi-user clinical environments
- [ ] Add model retraining pipeline when new data is available
- [ ] Build a REST API endpoint for integration with hospital systems
- [ ] Add DICOM / HL7 medical record import support

---

## ⚠️ Disclaimer

This tool is built for **educational purposes** as part of an AI Application Lab project. It is **not a medical device** and should not be used as a substitute for professional medical diagnosis or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Divyansh Singh**

[![GitHub](https://img.shields.io/badge/GitHub-divyansh2453-181717?style=flat-square&logo=github)](https://github.com/divyansh2453)
[![Live App](https://img.shields.io/badge/Live%20App-health--disease--ai.streamlit.app-FF4B4B?style=flat-square&logo=streamlit)](https://health-disease-ai.streamlit.app)

---

<div align="center">

Built with ❤️ as part of AI Application Lab — 2026

⭐ Star this repo if you found it useful!

</div>
