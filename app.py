import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FraudGuard AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    
    .main-title {
        text-align: center; font-size: 3.5rem; font-weight: 900; margin-bottom: 0px;
    }
    .gradient-text {
        font-family: 'Inter', sans-serif;
        background: -webkit-linear-gradient(45deg, #FF416C, #FF4B2B);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
    }
    .sub-title { text-align: center; font-size: 1.2rem; color: #a0aec0; margin-top: -10px; margin-bottom: 30px; }
    
    div[data-testid="metric-container"] {
        background-color: #1E293B; border: 1px solid #334155; padding: 15px; border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown('<div class="main-title">🛡️ <span class="gradient-text">FinSecure ML</span> 🛡️</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enterprise-Grade Credit Card Fraud Detection using XGBoost</div>', unsafe_allow_html=True)

# --- DATA LOADING & MOCK GENERATOR ---
@st.cache_data
def load_or_generate_data():
    try:
        df = pd.read_csv('creditcard.csv')
        source = "Real Kaggle Dataset"
    except FileNotFoundError:
        np.random.seed(42)
        n_samples = 10000
        data = {
            'Time': np.random.uniform(0, 172792, n_samples),
            'V1': np.random.normal(0, 2, n_samples),
            'V2': np.random.normal(0, 1.5, n_samples),
            'V3': np.random.normal(0, 1.5, n_samples),
            'Amount': np.random.exponential(50, n_samples),
        }
        df = pd.DataFrame(data)
        df['Class'] = 0
        fraud_indices = np.random.choice(n_samples, int(n_samples * 0.02), replace=False)
        df.loc[fraud_indices, 'Class'] = 1
        df.loc[fraud_indices, 'Amount'] = df.loc[fraud_indices, 'Amount'] * np.random.uniform(5, 10)
        df.loc[fraud_indices, 'V1'] = df.loc[fraud_indices, 'V1'] - 5
        source = "Auto-Generated Mock Dataset (creditcard.csv not found)"
    return df, source

df, data_source = load_or_generate_data()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.markdown("<h2 style='text-align: center;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
    st.info(f"📁 **Data Source:**\n{data_source}")
    
    st.divider()
    run_training = st.button("🚀 Train XGBoost Model", type="primary", use_container_width=True)
    
    st.markdown("### 🔍 Live Transaction Tester")
    test_amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
    test_v1 = st.slider("Anomaly Factor (V1)", min_value=-10.0, max_value=10.0, value=0.0)
    test_btn = st.button("Detect Fraud", use_container_width=True)

# --- MAIN DASHBOARD: EDA ---
st.markdown("### 📊 Dataset Overview")
col1, col2, col3 = st.columns(3)
total_tx = len(df)
fraud_tx = len(df[df['Class'] == 1])
fraud_rate = (fraud_tx / total_tx) * 100

col1.metric("Total Transactions Analyzed", f"{total_tx:,}")
col2.metric("Detected Fraud Cases", f"{fraud_tx:,}", delta_color="inverse")
col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")

fig1 = px.histogram(df, x="Amount", color="Class", nbins=50, log_y=True, 
                    title="Transaction Amount Distribution (Log Scale)",
                    color_discrete_map={0: '#10b981', 1: '#ef4444'})
fig1.update_layout(template='plotly_dark')
st.plotly_chart(fig1, use_container_width=True)

# --- AI TRAINING & EVALUATION ---
if run_training:
    st.divider()
    st.markdown("### 🧠 Model Training & Analytics (XGBoost)")
    
    with st.spinner("Training advanced XGBoost Classifier..."):
        # Preprocessing - FIXED SCALER BUG HERE
        X = df[['Time', 'V1', 'V2', 'V3', 'Amount']].copy()
        y = df['Class']
        
        scaler = StandardScaler()
        # Scale both columns at the same time so it remembers both
        X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        
        st.success("✅ Model Training Complete!")
        
        m1, m2 = st.columns(2)
        
        with m1:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Reds', 
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=['Legitimate', 'Fraud'], y=['Legitimate', 'Fraud'])
            fig_cm.update_layout(template='plotly_dark')
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with m2:
            st.markdown("#### ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (area = {roc_auc:.2f})', line=dict(color='cyan', width=2)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color='gray', dash='dash')))
            fig_roc.update_layout(template='plotly_dark', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
            st.plotly_chart(fig_roc, use_container_width=True)

# --- LIVE TESTER ---
if test_btn:
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train the model first by clicking 'Train XGBoost Model' in the sidebar.")
    else:
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        
        test_data = pd.DataFrame({'Time': [80000], 'V1': [test_v1], 'V2': [0.0], 'V3': [0.0], 'Amount': [test_amount]})
        test_data[['Time', 'Amount']] = scaler.transform(test_data[['Time', 'Amount']])
        
        prediction = model.predict(test_data)[0]
        confidence = model.predict_proba(test_data)[0][prediction] * 100
        
        st.divider()
        st.markdown("### 🔔 Transaction Alert System")
        
        # --- PREMIUM NEON RESULT CARDS ---
        if prediction == 1:
            fraud_html = f"""
            <div style="background: linear-gradient(145deg, #2a0808, #1a0505); border: 2px solid #ef4444; border-radius: 15px; padding: 30px; text-align: center; box-shadow: 0 0 30px rgba(239, 68, 68, 0.5);">
                <h1 style="color: #ef4444; margin-bottom: 5px; font-size: 45px; text-shadow: 0 0 10px rgba(239,68,68,0.8);">🚨 FRAUD DETECTED 🚨</h1>
                <h3 style="color: #fca5a5; margin-top: 0px; font-weight: 400;">Transaction Blocked Instantly</h3>
                <hr style="border-color: #7f1d1d; width: 50%;">
                <div style="font-size: 28px; color: white; margin: 20px 0;">
                    Threat Confidence: <strong style="color: #ef4444;">{confidence:.2f}%</strong>
                </div>
                <div style="background: rgba(0,0,0,0.5); padding: 10px; border-radius: 8px; display: inline-block; color: #9ca3af;">
                    <strong>Analyzed Data:</strong> Amount = ${test_amount} | Anomaly Factor = {test_v1}
                </div>
            </div>
            """
            st.markdown(fraud_html, unsafe_allow_html=True)
            
        else:
            safe_html = f"""
            <div style="background: linear-gradient(145deg, #022c22, #064e3b); border: 2px solid #10b981; border-radius: 15px; padding: 30px; text-align: center; box-shadow: 0 0 30px rgba(16, 185, 129, 0.4);">
                <h1 style="color: #10b981; margin-bottom: 5px; font-size: 45px; text-shadow: 0 0 10px rgba(16,185,129,0.8);">✅ TRANSACTION APPROVED ✅</h1>
                <h3 style="color: #6ee7b7; margin-top: 0px; font-weight: 400;">Payment Processed Successfully</h3>
                <hr style="border-color: #065f46; width: 50%;">
                <div style="font-size: 28px; color: white; margin: 20px 0;">
                    Safety Score: <strong style="color: #10b981;">{confidence:.2f}%</strong>
                </div>
                <div style="background: rgba(0,0,0,0.4); padding: 10px; border-radius: 8px; display: inline-block; color: #a7f3d0;">
                    <strong>Analyzed Data:</strong> Amount = ${test_amount} | Anomaly Factor = {test_v1}
                </div>
            </div>
            """
            st.markdown(safe_html, unsafe_allow_html=True)