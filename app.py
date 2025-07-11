import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os

# Configuration
st.set_page_config(page_title="Network Anomaly Detection", layout="wide")

# Title
st.title("Network Anomaly Detection System")
st.markdown("""
Detect unusual patterns in network traffic that may indicate security breaches
""")

# Sidebar for model selection
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox(
    "Select Detection Model",
    ("Isolation Forest", "Autoencoder")
)

# Load preprocessing objects
@st.cache_resource
def load_preprocessing():
    encoder = joblib.load('encoder.joblib')
    scaler = joblib.load('scaler.joblib')
    return encoder, scaler

encoder, scaler = load_preprocessing()

# Load selected model
@st.cache_resource
def load_selected_model(choice):
    if choice == "Isolation Forest":
        model = joblib.load('isolation_forest.joblib')
        return model, None
    else:
        model = load_model('autoencoder.h5')
        threshold = joblib.load('ae_threshold.joblib')
        return model, threshold

model, threshold = load_selected_model(model_choice)

# Sample data from KDD dataset (corrected)
SAMPLE_DATA = {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 215,
    "dst_bytes": 45076,
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 0,
    "num_failed_logins": 0,
    "logged_in": 1,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0,
    "num_file_creations": 0,
    "num_shells": 0,
    "num_access_files": 0,
    "is_host_login": 0,
    "is_guest_login": 0,
    "count": 2,
    "srv_count": 2,
    "serror_rate": 0.0,
    "srv_serror_rate": 0.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0,
    "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0,
    "dst_host_count": 25,
    "dst_host_srv_count": 25,
    "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 1.0,
    "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0
}

# Input form
st.header("Network Traffic Input")
col1, col2 = st.columns(2)

with col1:
    # Numerical features
    duration = st.number_input("Duration", value=SAMPLE_DATA["duration"])
    src_bytes = st.number_input("Source Bytes", value=SAMPLE_DATA["src_bytes"])
    dst_bytes = st.number_input("Destination Bytes", value=SAMPLE_DATA["dst_bytes"])
    count = st.number_input("Count", value=SAMPLE_DATA["count"])
    srv_count = st.number_input("Service Count", value=SAMPLE_DATA["srv_count"])

with col2:
    # Categorical features
    protocol_type = st.selectbox(
        "Protocol Type",
        ("tcp", "udp", "icmp"),
        index=0
    )
    service = st.selectbox(
        "Service",
        ("http", "smtp", "ftp", "other"),
        index=0
    )
    flag = st.selectbox(
        "Flag",
        ("SF", "S0", "S1", "REJ", "RSTO"),
        index=0
    )

# Prediction function
def predict_anomaly(input_data):
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # Preprocess
    categorical_cols = ['protocol_type', 'service', 'flag']
    numerical_cols = [col for col in df.columns if col not in categorical_cols + ['label', 'is_attack']]
    
    X_cat = encoder.transform(df[categorical_cols])
    X_num = scaler.transform(df[numerical_cols])
    X_processed = np.hstack((X_num, X_cat))
    
    # Predict based on model
    if model_choice == "Isolation Forest":
        prediction = (model.predict(X_processed) == -1).astype(int)[0]
        score = model.decision_function(X_processed)[0]
    else:
        reconstruction = model.predict(X_processed)
        mse = np.mean(np.square(X_processed - reconstruction))
        prediction = int(mse > threshold)
        score = mse
    
    return prediction, score

# Make prediction
if st.button("Detect Anomaly"):
    input_data = {
        "duration": duration,
        "protocol_type": protocol_type,
        "service": service,
        "flag": flag,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 1,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": count,
        "srv_count": srv_count,
        "serror_rate": 0.0,
        "srv_serror_rate": 0.0,
        "rerror_rate": 0.0,
        "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0,
        "dst_host_count": 25,
        "dst_host_srv_count": 25,
        "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 1.0,
        "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    }
    
    prediction, score = predict_anomaly(input_data)
    
    if prediction == 1:
        st.error(f"ANOMALY DETECTED (Score: {score:.4f})")
        st.markdown("""
        **Potential security issues detected:**  
        - Possible intrusion attempt  or
        - Suspicious network behavior  or
        - System malfunction
        """)
    else:
        st.success(f"Normal Traffic (Score: {score:.4f})")
        st.markdown("Network traffic appears normal")

# Dataset information
st.sidebar.header("Dataset Information")
st.sidebar.markdown("""
**KDD Cup 1999 Data**  
Corrected dataset available on [Kaggle](https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data)  
Contains network intrusion detection data  
- 4.9 million connections  
- 41 features  
- 23 attack types
""")

# Run instructions
st.sidebar.header("How to Use")
st.sidebar.markdown("""
1. Select detection model  
2. Enter/Modify network traffic features  
3. Click "Detect Anomaly"  
4. View results  
*Default values show sample normal traffic*
""")