# 🛡️ FraudGuard AI - Financial Anomaly Detection

![XGBoost](https://img.shields.io/badge/Model-XGBoost-blue.svg)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458.svg)

## 🚀 Project Overview
FraudGuard AI is a high-performance predictive model built to detect fraudulent credit card transactions in real-time. Designed with cybersecurity in mind, it utilizes the powerful XGBoost algorithm to handle highly imbalanced financial datasets.

## Live App Link:

https://fraudguard-ai-anomaly-detection.streamlit.app/

## Screenshots

## Main Dashboard
<img width="1919" height="1079" alt="Screenshot 2026-03-24 121410" src="https://github.com/user-attachments/assets/b29e76f8-917f-4ff2-b098-034e01668013" />

## Safe Transaction
<img width="1914" height="920" alt="Screenshot 2026-03-24 121542" src="https://github.com/user-attachments/assets/cb083633-b4f5-4200-9e2f-5c570f24cec6" />

## Fraud Transaction
<img width="1917" height="922" alt="Screenshot 2026-03-24 121704" src="https://github.com/user-attachments/assets/9de3d8e1-aa3b-40fa-86d3-d2d4d28c476f" />

*Developed as part of the AI & ML Internship at Elevate Labs.*

## 🧠 System Architecture & Methodology
1. **Data Simulation:** Auto-generates realistic financial mock data if external datasets are missing.
2. **Preprocessing:** Standardizes features using `StandardScaler`.
3. **Handling Imbalance:** Integrates native XGBoost `scale_pos_weight` tuning.
4. **Evaluation:** Generates dynamic Confusion Matrices and ROC-AUC curves.

## 🔥 Key Features
* **Gradient Boosting:** Lightning-fast inference using XGBoost.
* **Real-time Alert System:** Instant visual triggers for blocked fraudulent transactions.
* **Live Transaction Tester:** Interactive sliders to inject custom anomalies.

## ⚙️ Installation & Usage
git clone https://github.com/your-username/FraudGuard-AI-Anomaly-Detection.git
cd FraudGuard-AI-Anomaly-Detection
pip install -r requirements.txt
streamlit run app.py

## 👨‍💻 Author
**Md Salman Farsi**
* **Role:** AI & ML Intern @ Elevate Labs | B.Tech CSE (AI & ML)
* **Portfolio:** [mdsalmanfarsi.io](https://mdsalmanfarsi692004-svg.github.io/portfolio/)
* **Email:** [mdsalmanfarsi692004@gmail.com](mailto:mdsalmanfarsi692004@gmail.com)
* **GitHub:** [https://github.com/mdsalmanfarsi692004-svg]
* **LinkedIn:** [www.linkedin.com/in/md-salman-farsi-data-analyst]
