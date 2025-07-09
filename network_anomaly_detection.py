import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class NetworkAnomalyDetector:
    def __init__(self):
        self.scaler=StandardScaler()
        self.label_encoders={}
        self.isolation_forest=None
        self.autoencoder=None
        self.encoder=None
        self.feature_names=None
        
    def load_and_preprocess_data(self, file_path=None):
        """
        Load and preprocess KDD Cup 1999 dataset
        """
        # KDD Cup 1999 column names
        column_names=[
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
        ]
        
        # If no file path provided, create sample data for demonstration
        if file_path is None:
            print("Creating sample data for demonstration...")
            data=self._create_sample_data()
        else:
            print(f"Loading data from {file_path}...")
            data=pd.read_csv(file_path, names=column_names)
        
        print(f"Dataset shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        
        # Display basic statistics
        print("\nLabel distribution:")
        print(data['label'].value_counts())
        
        # Create binary labels: normal vs anomaly
        data['is_anomaly']=(data['label'] != 'normal.').astype(int)
        
        return data
    
    def _create_sample_data(self):
        """
        Create sample network traffic data for demonstration
        """
        np.random.seed(42)
        n_samples=10000
        
        # Generate normal traffic patterns
        normal_data={
            'duration': np.random.exponential(2, n_samples),
            'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples, p=[0.7, 0.2, 0.1]),
            'service': np.random.choice(['http', 'ftp', 'smtp', 'dns', 'ssh'], n_samples),
            'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTR'], n_samples, p=[0.7, 0.15, 0.1, 0.05]),
            'src_bytes': np.random.lognormal(5, 2, n_samples),
            'dst_bytes': np.random.lognormal(4, 2, n_samples),
            'land': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
            'wrong_fragment': np.random.poisson(0.1, n_samples),
            'urgent': np.random.poisson(0.05, n_samples),
            'hot': np.random.poisson(0.2, n_samples),
            'num_failed_logins': np.random.poisson(0.1, n_samples),
            'logged_in': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'num_compromised': np.random.poisson(0.05, n_samples),
            'root_shell': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'su_attempted': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
            'num_root': np.random.poisson(0.1, n_samples),
            'num_file_creations': np.random.poisson(0.3, n_samples),
            'num_shells': np.random.poisson(0.05, n_samples),
            'num_access_files': np.random.poisson(0.2, n_samples),
            'num_outbound_cmds': np.random.poisson(0.01, n_samples),
            'is_host_login': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
            'is_guest_login': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'count': np.random.poisson(10, n_samples),
            'srv_count': np.random.poisson(8, n_samples),
            'serror_rate': np.random.beta(1, 10, n_samples),
            'srv_serror_rate': np.random.beta(1, 10, n_samples),
            'rerror_rate': np.random.beta(1, 15, n_samples),
            'srv_rerror_rate': np.random.beta(1, 15, n_samples),
            'same_srv_rate': np.random.beta(5, 2, n_samples),
            'diff_srv_rate': np.random.beta(1, 5, n_samples),
            'srv_diff_host_rate': np.random.beta(1, 8, n_samples),
            'dst_host_count': np.random.poisson(20, n_samples),
            'dst_host_srv_count': np.random.poisson(15, n_samples),
            'dst_host_same_srv_rate': np.random.beta(5, 2, n_samples),
            'dst_host_diff_srv_rate': np.random.beta(1, 5, n_samples),
            'dst_host_same_src_port_rate': np.random.beta(3, 3, n_samples),
            'dst_host_srv_diff_host_rate': np.random.beta(1, 8, n_samples),
            'dst_host_serror_rate': np.random.beta(1, 10, n_samples),
            'dst_host_srv_serror_rate': np.random.beta(1, 10, n_samples),
            'dst_host_rerror_rate': np.random.beta(1, 15, n_samples),
            'dst_host_srv_rerror_rate': np.random.beta(1, 15, n_samples),
            'label': ['normal.'] * n_samples
        }
        
        # Generate anomalous traffic patterns (10% of data)
        n_anomalies=int(n_samples * 0.1)
        anomaly_types=['dos.', 'probe.', 'r2l.', 'u2r.']
        
        anomaly_data={
            'duration': np.random.exponential(10, n_anomalies),  # Longer durations
            'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_anomalies),
            'service': np.random.choice(['http', 'ftp', 'smtp', 'dns', 'ssh'], n_anomalies),
            'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTR'], n_anomalies),
            'src_bytes': np.random.lognormal(8, 3, n_anomalies),  # Larger bytes
            'dst_bytes': np.random.lognormal(7, 3, n_anomalies),  # Larger bytes
            'land': np.random.choice([0, 1], n_anomalies, p=[0.8, 0.2]),  # More land attacks
            'wrong_fragment': np.random.poisson(2, n_anomalies),  # More fragments
            'urgent': np.random.poisson(1, n_anomalies),  # More urgent packets
            'hot': np.random.poisson(3, n_anomalies),  # More hot indicators
            'num_failed_logins': np.random.poisson(2, n_anomalies),  # More failed logins
            'logged_in': np.random.choice([0, 1], n_anomalies, p=[0.7, 0.3]),
            'num_compromised': np.random.poisson(1, n_anomalies),  # More compromised
            'root_shell': np.random.choice([0, 1], n_anomalies, p=[0.7, 0.3]),  # More root shells
            'su_attempted': np.random.choice([0, 1], n_anomalies, p=[0.8, 0.2]),  # More su attempts
            'num_root': np.random.poisson(2, n_anomalies),  # More root accesses
            'num_file_creations': np.random.poisson(5, n_anomalies),  # More file creations
            'num_shells': np.random.poisson(1, n_anomalies),  # More shells
            'num_access_files': np.random.poisson(3, n_anomalies),  # More file accesses
            'num_outbound_cmds': np.random.poisson(0.5, n_anomalies),
            'is_host_login': np.random.choice([0, 1], n_anomalies, p=[0.95, 0.05]),
            'is_guest_login': np.random.choice([0, 1], n_anomalies, p=[0.8, 0.2]),
            'count': np.random.poisson(50, n_anomalies),  # Higher counts
            'srv_count': np.random.poisson(40, n_anomalies),  # Higher service counts
            'serror_rate': np.random.beta(3, 5, n_anomalies),  # Higher error rates
            'srv_serror_rate': np.random.beta(3, 5, n_anomalies),
            'rerror_rate': np.random.beta(2, 3, n_anomalies),
            'srv_rerror_rate': np.random.beta(2, 3, n_anomalies),
            'same_srv_rate': np.random.beta(2, 5, n_anomalies),
            'diff_srv_rate': np.random.beta(3, 2, n_anomalies),
            'srv_diff_host_rate': np.random.beta(3, 3, n_anomalies),
            'dst_host_count': np.random.poisson(100, n_anomalies),  # Higher host counts
            'dst_host_srv_count': np.random.poisson(80, n_anomalies),
            'dst_host_same_srv_rate': np.random.beta(2, 5, n_anomalies),
            'dst_host_diff_srv_rate': np.random.beta(3, 2, n_anomalies),
            'dst_host_same_src_port_rate': np.random.beta(2, 5, n_anomalies),
            'dst_host_srv_diff_host_rate': np.random.beta(3, 3, n_anomalies),
            'dst_host_serror_rate': np.random.beta(3, 5, n_anomalies),
            'dst_host_srv_serror_rate': np.random.beta(3, 5, n_anomalies),
            'dst_host_rerror_rate': np.random.beta(2, 3, n_anomalies),
            'dst_host_srv_rerror_rate': np.random.beta(2, 3, n_anomalies),
            'label': np.random.choice(anomaly_types, n_anomalies)
        }
        
        # Combine normal and anomalous data
        combined_data={}
        for key in normal_data.keys():
            combined_data[key]=np.concatenate([normal_data[key], anomaly_data[key]])
        
        # Create DataFrame and shuffle
        df=pd.DataFrame(combined_data)
        df=df.sample(frac=1).reset_index(drop=True)
        
        return df
    
    def preprocess_features(self, data):
        """
        Preprocess features for machine learning
        """
        # Separate features and labels
        X=data.drop(['label', 'is_anomaly'], axis=1)
        y=data['is_anomaly']
        
        # Handle categorical variables
        categorical_cols=['protocol_type', 'service', 'flag']
        
        for col in categorical_cols:
            if col in X.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col]=LabelEncoder()
                    X[col]=self.label_encoders[col].fit_transform(X[col])
                else:
                    X[col]=self.label_encoders[col].transform(X[col])
        
        # Store feature names
        self.feature_names=X.columns.tolist()
        
        # Scale features
        X_scaled=self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_isolation_forest(self, X, contamination=0.1):
        """
        Train Isolation Forest for anomaly detection
        """
        print("Training Isolation Forest...")
        self.isolation_forest=IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.isolation_forest.fit(X)
        print("Isolation Forest training completed!")
        
    def build_autoencoder(self, input_dim, encoding_dim=32):
        """
        Build and compile autoencoder model
        """
        print("Building Autoencoder...")
        
        # Input layer
        input_layer=Input(shape=(input_dim,))
        
        # Encoder
        encoded=Dense(64, activation='relu')(input_layer)
        encoded=Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded=Dense(64, activation='relu')(encoded)
        decoded=Dense(input_dim, activation='linear')(decoded)
        
        # Autoencoder model
        autoencoder=Model(input_layer, decoded)
        
        # Encoder model
        encoder=Model(input_layer, encoded)
        
        # Compile
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), 
                           loss='mse', 
                           metrics=['mae'])
        
        return autoencoder, encoder
    
    def train_autoencoder(self, X_normal, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train autoencoder on normal data only
        """
        print("Training Autoencoder...")
        
        # Build autoencoder
        self.autoencoder, self.encoder=self.build_autoencoder(X_normal.shape[1])
        
        # Train on normal data only
        history=self.autoencoder.fit(
            X_normal, X_normal,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True,
            verbose=1
        )
        
        print("Autoencoder training completed!")
        return history
    
    def predict_isolation_forest(self, X):
        """
        Predict anomalies using Isolation Forest
        """
        # Predict (-1 for anomalies, 1 for normal)
        predictions=self.isolation_forest.predict(X)
        
        # Convert to binary (1 for anomalies, 0 for normal)
        binary_predictions=(predictions == -1).astype(int)
        
        # Get anomaly scores
        anomaly_scores=self.isolation_forest.decision_function(X)
        
        return binary_predictions, anomaly_scores
    
    def predict_autoencoder(self, X, threshold=None):
        """
        Predict anomalies using Autoencoder
        """
        # Reconstruct data
        reconstructed=self.autoencoder.predict(X)
        
        # Calculate reconstruction error
        reconstruction_error=np.mean(np.square(X - reconstructed), axis=1)
        
        # Set threshold if not provided
        if threshold is None:
            threshold=np.percentile(reconstruction_error, 95)
        
        # Binary predictions
        binary_predictions=(reconstruction_error > threshold).astype(int)
        
        return binary_predictions, reconstruction_error, threshold
    
    def evaluate_model(self, y_true, y_pred, scores, model_name):
        """
        Evaluate model performance
        """
        print(f"\n{model_name} Performance:")
        print("=" * 50)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        
        # Confusion matrix
        cm=confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        # ROC curve
        fpr, tpr, _=roc_curve(y_true, scores)
        auc_score=roc_auc_score(y_true, scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve')
        plt.legend()
        plt.show()
        
        return auc_score
    
    def feature_importance_isolation_forest(self, X, feature_names, n_samples=1000):
        """
        Calculate feature importance for Isolation Forest
        """
        print("Calculating feature importance...")
        
        # Sample data for faster computation
        if len(X) > n_samples:
            indices=np.random.choice(len(X), n_samples, replace=False)
            X_sample=X[indices]
        else:
            X_sample=X
        
        # Calculate baseline scores
        baseline_scores=self.isolation_forest.decision_function(X_sample)
        
        # Calculate importance by permuting each feature
        importances=[]
        for i in range(X_sample.shape[1]):
            X_permuted=X_sample.copy()
            X_permuted[:, i]=np.random.permutation(X_permuted[:, i])
            
            permuted_scores=self.isolation_forest.decision_function(X_permuted)
            importance=np.mean(np.abs(baseline_scores - permuted_scores))
            importances.append(importance)
        
        # Create feature importance DataFrame
        importance_df=pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature')
        plt.title('Isolation Forest - Top 15 Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def visualize_anomalies(self, X, y_true, y_pred, scores, model_name):
        """
        Visualize anomaly detection results
        """
        # Create 2D projection using first two principal components
        from sklearn.decomposition import PCA
        
        pca=PCA(n_components=2)
        X_2d=pca.fit_transform(X)
        
        # Create scatter plot
        plt.figure(figsize=(12, 5))
        
        # Plot 1: True vs Predicted
        plt.subplot(1, 2, 1)
        colors=['blue' if pred == 0 else 'red' for pred in y_pred]
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.6, s=20)
        plt.title(f'{model_name} - Predicted Anomalies\n(Blue: Normal, Red: Anomaly)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        
        # Plot 2: Anomaly scores
        plt.subplot(1, 2, 2)
        scatter=plt.scatter(X_2d[:, 0], X_2d[:, 1], c=scores, alpha=0.6, s=20, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f'{model_name} - Anomaly Scores')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, file_path=None):
        """
        Run complete anomaly detection analysis
        """
        print("Starting Network Traffic Anomaly Detection Analysis")
        print("=" * 60)
        
        # Load and preprocess data
        data=self.load_and_preprocess_data(file_path)
        X, y=self.preprocess_features(data)
        
        # Split data
        X_train, X_test, y_train, y_test=train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Get normal data for autoencoder training
        X_normal=X_train[y_train == 0]
        
        print(f"\nDataset split:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Normal training samples: {X_normal.shape[0]}")
        
        # 1. ISOLATION FOREST
        print("\n" + "="*60)
        print("ISOLATION FOREST ANALYSIS")
        print("="*60)
        
        self.train_isolation_forest(X_train, contamination=0.1)
        
        # Predict on test set
        if_predictions, if_scores=self.predict_isolation_forest(X_test)
        
        # Evaluate Isolation Forest
        if_auc=self.evaluate_model(y_test, if_predictions, -if_scores, "Isolation Forest")
        
        # Feature importance
        if_importance=self.feature_importance_isolation_forest(X_test, self.feature_names)
        
        # Visualize results
        self.visualize_anomalies(X_test, y_test, if_predictions, -if_scores, "Isolation Forest")
        
        # 2. AUTOENCODER
        print("\n" + "="*60)
        print("AUTOENCODER ANALYSIS")
        print("="*60)
        
        # Train autoencoder
        ae_history=self.train_autoencoder(X_normal, epochs=50, batch_size=64)
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(ae_history.history['loss'], label='Training Loss')
        plt.plot(ae_history.history['val_loss'], label='Validation Loss')
        plt.title('Autoencoder Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(ae_history.history['mae'], label='Training MAE')
        plt.plot(ae_history.history['val_mae'], label='Validation MAE')
        plt.title('Autoencoder Training MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Predict on test set
        ae_predictions, ae_scores, ae_threshold=self.predict_autoencoder(X_test)
        
        print(f"Autoencoder threshold: {ae_threshold:.4f}")
        
        # Evaluate Autoencoder
        ae_auc=self.evaluate_model(y_test, ae_predictions, ae_scores, "Autoencoder")
        
        # Visualize results
        self.visualize_anomalies(X_test, y_test, ae_predictions, ae_scores, "Autoencoder")
        
        # 3. COMPARISON
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_df=pd.DataFrame({
            'Model': ['Isolation Forest', 'Autoencoder'],
            'AUC Score': [if_auc, ae_auc],
            'Precision (Anomaly)': [
                classification_report(y_test, if_predictions, output_dict=True)['1']['precision'],
                classification_report(y_test, ae_predictions, output_dict=True)['1']['precision']
            ],
            'Recall (Anomaly)': [
                classification_report(y_test, if_predictions, output_dict=True)['1']['recall'],
                classification_report(y_test, ae_predictions, output_dict=True)['1']['recall']
            ],
            'F1-Score (Anomaly)': [
                classification_report(y_test, if_predictions, output_dict=True)['1']['f1-score'],
                classification_report(y_test, ae_predictions, output_dict=True)['1']['f1-score']
            ]
        })
        
        print("\nModel Comparison:")
        print(comparison_df.round(4))
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        metrics=['AUC Score', 'Precision (Anomaly)', 'Recall (Anomaly)', 'F1-Score (Anomaly)']
        x=np.arange(len(metrics))
        width=0.35
        
        plt.bar(x - width/2, comparison_df.iloc[0, 1:], width, label='Isolation Forest', alpha=0.8)
        plt.bar(x + width/2, comparison_df.iloc[1, 1:], width, label='Autoencoder', alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # 4. ANOMALY ANALYSIS
        print("\n" + "="*60)
        print("ANOMALY ANALYSIS")
        print("="*60)
        
        # Analyze detected anomalies
        anomaly_indices=np.where(if_predictions == 1)[0]
        
        if len(anomaly_indices) > 0:
            print(f"\nIsolation Forest detected {len(anomaly_indices)} anomalies")
            
            # Show some examples of detected anomalies
            anomaly_samples=X_test[anomaly_indices[:5]]  # First 5 anomalies
            
            print("\nTop 5 detected anomalies (feature values):")
            for i, sample in enumerate(anomaly_samples):
                print(f"\nAnomaly {i+1}:")
                for j, feature_name in enumerate(self.feature_names[:10]):  # Show first 10 features
                    print(f"  {feature_name}: {sample[j]:.4f}")
        
        return {
            'isolation_forest_auc': if_auc,
            'autoencoder_auc': ae_auc,
            'comparison_df': comparison_df,
            'feature_importance': if_importance
        }

# Main execution
if __name__ == "__main__":
    # Initialize detector
    detector=NetworkAnomalyDetector()
    
    
    results=detector.run_complete_analysis()
    
    print("\nAnalysis completed successfully!")
    print("Key findings:")
    print(f"- Isolation Forest AUC: {results['isolation_forest_auc']:.4f}")
    print(f"- Autoencoder AUC: {results['autoencoder_auc']:.4f}")
    
    # Save results
    results['comparison_df'].to_csv('model_comparison.csv', index=False)
    results['feature_importance'].to_csv('feature_importance.csv', index=False)
    
    print("\nResults saved to:")
    print("- model_comparison.csv")
    print("- feature_importance.csv")
