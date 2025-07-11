# Network Anomaly Detection using Unsupervised Learning

## Project Overview

This project implements unsupervised learning techniques to detect unusual patterns and anomalies in network traffic data that could indicate potential security breaches or system malfunctions. The system uses the KDD Cup 1999 dataset and employs two powerful anomaly detection algorithms: Isolation Forest (tree-based) and Autoencoder (neural network-based reconstruction).

## Problem Statement

Using unsupervised learning techniques such as Isolation Forests and Autoencoders to detect unusual patterns or anomalies in network traffic data, which could indicate potential security breaches or system malfunctions using the KDD Cup 1999 dataset.

## Dataset Information

**KDD Cup 1999 Network Intrusion Detection Dataset**
- Size: ~4.9 million network connections
- Features: 41 attributes per connection
- Attack Types: 23 different types of network attacks
- Classes: Normal traffic vs. Attack traffic
- Source: [Kaggle - KDD Cup 1999 Data](https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data)

The dataset includes basic features (duration, protocol type, service, flag, bytes transferred), content features (failed logins, compromised conditions, root shell access), traffic features (connection counts, error rates, service patterns), and host-based features (destination host statistics and patterns).

## Project Structure

```
├── preprocessing.ipynb          # Data preprocessing and feature engineering
├── isolation_forest.ipynb       # Isolation Forest implementation
├── autoencoder.ipynb           # Autoencoder implementation  
├── data-visualization.ipynb    # Comprehensive data analysis and visualization
├── app.py                      # Streamlit web application
├── dataset/                    # KDD Cup 1999 dataset files                   
├── isolation_forest.joblib     # Trained model files
├── autoencoder.h5              # Trained model files
├── encoder.joblib              # Trained model files
├── scaler.joblib               # Trained model files
└── ae_threshold.joblib         # Trained model files
└── README.md
```

## Implementation Details

### Data Preprocessing
The preprocessing pipeline handles 41 network traffic features through categorical encoding (one-hot encoding for protocol_type, service, flag), normalization using StandardScaler, binary classification conversion, and stratified 70-30 train-test splitting with memory optimization.

### Isolation Forest
This tree-based anomaly detection algorithm trains on normal traffic only using unsupervised learning. Key parameters include contamination=0.2, n_estimators=300, and max_samples='auto'. Evaluation metrics include ROC-AUC, Average Precision, and confusion matrix analysis.

### Autoencoder
The neural network architecture consists of an encoder (128 → 64 neurons with ReLU activation) and decoder (64 → 128 → input size with linear output). Training uses reconstruction loss (MSE) on normal traffic, with anomaly detection based on high reconstruction error using the 95th percentile threshold.

### Data Visualization
Comprehensive analysis includes class distribution analysis, feature correlation heatmaps, PCA visualization in 2D and 3D, interactive Plotly charts, feature importance analysis, and distribution analysis with KDE plots.

### Web Application
The Streamlit-based interface provides model selection between Isolation Forest and Autoencoder, real-time prediction capabilities, intuitive form-based input with sample data, and clear anomaly classification with confidence scores.

## Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn plotly streamlit joblib
```

### Installation and Usage

1. Clone the repository and navigate to the project directory
2. Download the KDD Cup 1999 dataset from Kaggle and place in the `dataset/` directory
3. Run preprocessing: `jupyter notebook preprocessing.ipynb`
4. Train models: `jupyter notebook isolation_forest.ipynb` and `jupyter notebook autoencoder.ipynb`
5. Explore visualizations: `jupyter notebook data-visualization.ipynb`
6. Launch web application: `streamlit run app.py`

## Model Performance

**Isolation Forest** employs unsupervised learning on normal traffic patterns with anomaly scoring based on isolation difficulty. It offers fast training, interpretable results, and efficient handling of high-dimensional data, making it suitable for real-time anomaly detection.

**Autoencoder** uses reconstruction-based learning on normal traffic with high reconstruction error indicating anomalies. It captures complex non-linear patterns through deep feature learning, making it ideal for detailed anomaly analysis.

## Security Applications

The system detects various network anomalies including DoS/DDoS attacks, port scanning, buffer overflow attempts, unauthorized access, data exfiltration, and system malfunctions.

## Key Features

- Unsupervised learning approach requiring no labeled attack data for training
- Dual algorithm comparison between Isolation Forest and Autoencoder
- Scalable processing with memory-efficient handling of large datasets
- Interactive web interface for user-friendly anomaly detection
- Comprehensive data exploration and visualization capabilities
- Production-ready with saved models and preprocessing pipelines

## Technical Stack

- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Machine Learning**: Isolation Forest, TensorFlow/Keras
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Framework**: Streamlit
- **Model Persistence**: Joblib, HDF5

## Future Enhancements

Potential improvements include real-time stream processing, ensemble methods combining multiple algorithms, advanced neural architectures (LSTM, GAN-based), domain-specific feature engineering, automated alert systems, and continuous learning capabilities.



---

**Note**: This project demonstrates the application of unsupervised learning techniques for cybersecurity purposes using the KDD Cup 1999 dataset. The implemented models serve as a foundation for advanced network intrusion detection systems.