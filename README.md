# 🫀 Heart Disease Risk Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-ff4b4b?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> An AI-powered web application that predicts heart disease risk using machine learning, built as part of the AI Application Lab course.

🌐 **Live App:** [health-disease-ai.streamlit.app](https://health-disease-ai.streamlit.app)

---

## 📌 About

This project presents a Heart Disease Risk Prediction Dashboard that leverages machine learning classification algorithms to predict the likelihood of heart disease in patients based on clinical features. The system supports both individual patient assessments and batch processing of multiple patients via CSV upload.

The project covers the full ML pipeline — from data exploration and model training in Google Colab, to deployment as a fully interactive web application using Streamlit.

---

## ✨ Features

### 🧍 Single Patient Mode
- Input form with sliders and dropdowns for all 13 clinical features
- Human-readable labels (e.g. "Typical Angina" instead of `0`)
- Personal risk gauge meter (0–100%)
- Color-coded result: 🔴 High Risk / 🟢 Low Risk
- Patient summary table

### 📋 Batch Upload Mode
- Upload a CSV file with multiple patients
- Predictions for all patients in one click
- Summary metrics: total, high risk, low risk, average confidence
- Batch risk gauge showing overall group risk level
- Interactive charts:
  - Risk distribution donut chart
  - Risk by age group histogram
  - Cholesterol vs Blood Pressure scatter plot
  - Prediction confidence distribution
- High-risk patients table
- Export as CSV or PDF report

### 📊 Always Visible
- Feature importance chart in sidebar (top 5 risk factors)
- Model info and usage instructions

---

## 🧠 Model

| Property | Value |
|---|---|
| Dataset | Cleveland Heart Disease (1025 patients) |
| Algorithm | Gradient Boosting Classifier (best of 6 compared) |
| Accuracy | ~85% |
| AUC Score | ~92% |
| Train/Test Split | 80% / 20% |
| Preprocessing | StandardScaler normalization |

### Models Compared

| Model | AUC |
|---|---|
| Gradient Boosting | ~92% ✅ Best |
| Random Forest | ~91% |
| Logistic Regression | ~90% |
| SVM | ~90% |
| KNN | ~85% |
| Decision Tree | ~76% |

---

## 🗂️ Project Structure

```
Health_Disease_Dashboard/
│
├── app.py                  ← Main Streamlit dashboard
├── requirements.txt        ← Python dependencies
├── generate_sample.py      ← Script to generate random test data
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

### 4. Generate test data (optional)
```bash
python generate_sample.py --n 20
```

Open your browser at `http://localhost:8501` 🎉

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

The model was trained on the **Cleveland Heart Disease Dataset** from the UCI Machine Learning Repository.

- 📥 Download: [Kaggle — Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- 1025 patients, 14 features, binary classification target
- Well balanced: 51.3% positive, 48.7% negative

### Feature Description

| Feature | Description |
|---|---|
| age | Age of the patient |
| sex | Sex (1 = Male, 0 = Female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mmHg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0–4) |
| thal | Thalassemia type (0–3) |
| target | Heart disease present (1 = Yes, 0 = No) |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| scikit-learn | Model training and evaluation |
| Streamlit | Web application framework |
| Plotly | Interactive visualizations |
| fpdf2 | PDF report generation |
| pandas / numpy | Data processing |
| Google Colab | Model training environment |
| Google Drive | Persistent model storage |

---

## 🧪 Generate Sample Data

Use the included script to generate fresh random patient data for testing:

```bash
# Default 20 patients
python generate_sample.py

# Custom count
python generate_sample.py --n 50

# Fixed seed for reproducibility
python generate_sample.py --n 20 --seed 42

# Custom output file
python generate_sample.py --n 100 --output my_patients.csv
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Divyansh Singh**
- GitHub: [@divyansh2453](https://github.com/divyansh2453)
- Live App: [health-disease-ai.streamlit.app](https://health-disease-ai.streamlit.app)

---

> Built with ❤️ as part of AI Application Lab — 2026
