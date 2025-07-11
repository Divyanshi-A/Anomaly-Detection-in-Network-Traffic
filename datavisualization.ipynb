# Data Visualization Notebook for Network Anomaly Detection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8')  # Updated style for newer matplotlib versions
sns.set_palette("husl")
sns.set_style("whitegrid")
%matplotlib inline

# Load preprocessed data
try:
    X = np.load('X_train.npy')
    y = np.load('y_train.npy')
except FileNotFoundError:
    raise FileNotFoundError("Preprocessed data files not found. Run preprocessing.ipynb first.")

# Generate proper feature names - 120 features as per preprocessing
num_features = X.shape[1]
feature_names = [f"feature_{i+1}" for i in range(num_features)]

# Create DataFrame for visualization
df = pd.DataFrame(X, columns=feature_names)
df['is_attack'] = y

# 1. Attack Class Distribution
plt.figure(figsize=(10,6))
attack_counts = df['is_attack'].value_counts()
plt.pie(attack_counts, labels=['Normal', 'Attack'], autopct='%1.1f%%',
        colors=['#4c72b0','#c44e52'], startangle=90)
plt.title('Network Traffic Class Distribution')
plt.show()

# 2. Feature Correlation Heatmap (Top 20)
plt.figure(figsize=(12,10))
corr_matrix = df.iloc[:,:20].corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Top 20 Features Correlation Heatmap')
plt.tight_layout()
plt.show()

# 3. PCA Visualization (2D and 3D)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# 2D PCA Plot
plt.figure(figsize=(10,8))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette={0:'blue',1:'red'}, alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D PCA of Network Traffic')
plt.legend(title='Class', labels=['Normal', 'Attack'])
plt.show()

# 3D Interactive PCA Plot
fig = px.scatter_3d(
    x=X_pca[:,0], y=X_pca[:,1], z=X_pca[:,2],
    color=y, color_continuous_scale=['blue', 'red'],
    labels={'color': 'Attack Status'},
    title='3D PCA of Network Traffic'
)
fig.update_layout(
    scene=dict(
        xaxis_title='PCA 1',
        yaxis_title='PCA 2', 
        zaxis_title='PCA 3'
    )
)
fig.show()

# 4. Feature Importance Analysis (Using PCA)
pca_loadings = pd.DataFrame(
    pca.components_[:3].T,
    columns=['PC1','PC2','PC3'],
    index=feature_names
)

# Top 10 features for each component
for i, col in enumerate(pca_loadings.columns):
    print(f"\nTop 10 features for {col}:")
    print(pca_loadings[col].abs().sort_values(ascending=False).head(10))

# 5. Attack Pattern Radar Chart (Interactive)
attack_features = df[df['is_attack']==1].iloc[:,:10].mean().values
normal_features = df[df['is_attack']==0].iloc[:,:10].mean().values

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=attack_features,
    theta=feature_names[:10],
    fill='toself',
    name='Attack Traffic'
))

fig.add_trace(go.Scatterpolar(
    r=normal_features,
    theta=feature_names[:10],
    fill='toself',
    name='Normal Traffic'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True)
    ),
    title='Feature Comparison: Attack vs Normal Traffic'
)
fig.show()

# 6. Feature Distribution Analysis
plt.figure(figsize=(14,6))
sns.kdeplot(df[df['is_attack']==0]['feature_1'], label='Normal', shade=True)
sns.kdeplot(df[df['is_attack']==1]['feature_1'], label='Attack', shade=True)
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Density')
plt.title('Feature 1 Distribution by Class')
plt.legend()
plt.show()

# 7. Parallel Coordinates Plot (Top 5 features)
top_features = pca_loadings['PC1'].abs().sort_values(ascending=False).head(5).index.tolist()
top_features.append('is_attack')

fig = px.parallel_coordinates(
    df[top_features].sample(1000),  # Sample for performance
    color='is_attack',
    color_continuous_scale=['blue', 'red'],
    labels={'is_attack': 'Attack Status'},
    title='Parallel Coordinates of Top 5 Discriminative Features'
)
fig.show()

# 8. Boxplot of Top Features
plt.figure(figsize=(12, 6))
top_5_features = pca_loadings['PC1'].abs().sort_values(ascending=False).head(5).index.tolist()
melted_df = pd.melt(df[top_5_features + ['is_attack']], id_vars='is_attack', 
                    var_name='feature', value_name='value')
sns.boxplot(x='feature', y='value', hue='is_attack', data=melted_df)
plt.title('Distribution of Top 5 Features by Class')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
