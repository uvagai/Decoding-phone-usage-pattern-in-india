#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[3]:


#load data
df=pd.read_csv("phone_usage_india.csv")
df


# In[4]:


# Check for missing values
missing_values = df.isnull().sum()

# Impute missing values: Numeric columns with median, Categorical with mode
for column in df.columns:
    if df[column].isnull().sum() > 0:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].median(), inplace=True)


# In[5]:


df.drop(columns=['User ID'], inplace=True)
df.info()


# In[6]:


df.describe()

# Standardize Formats
# Convert OS and Phone Brand to lowercase for consistency
df['OS'] = df['OS'].str.strip().str.lower()
df['Phone Brand'] = df['Phone Brand'].str.strip().str.title()

# In[ ]:





# In[8]:


# changing the software to single name
df.loc[df['Phone Brand'] == "Apple", 'OS'] = "iOS"
df.loc[df['Phone Brand'] != "Apple", 'OS'] = "Android"


# In[9]:


# verifying the changes
print(df.groupby(['Phone Brand', 'OS']).size())
print(df['OS'].value_counts())


# In[10]:


columns = ['Screen Time (hrs/day)', 'Data Usage (GB/month)', 'Calls Duration (mins/day)',
            'Number of Apps Installed', 'Social Media Time (hrs/day)', 'E-commerce Spend (INR/month)',
            'Streaming Time (hrs/day)', 'Gaming Time (hrs/day)', 'Monthly Recharge Cost (INR)']


# In[11]:


# checking for outliers

plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include=[np.number]), orient="v", width=0.7)
plt.xticks(rotation=45, ha="right", fontsize=12)  # Rotate and align right
plt.title("Boxplot of Numerical Features", fontsize=14)
plt.show()


# In[17]:


# Remove Outliers using IQR method
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df
    df = remove_outliers(df, columns)


# In[21]:


# List of numerical columns to check for outliers
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df_cleaned = remove_outliers(df, numeric_cols)


# In[39]:


# Checking results missing values and cleaned data shape
df_cleaned.info(), df_cleaned.head()


# In[41]:


fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
axes = axes.flatten()
for i, col in enumerate(columns):
    sns.histplot(df[col], kde=True, color=sns.color_palette("Set1", len(columns))[i], ax=axes[i])
    axes[i].set_title(f"Distribution of {col}", fontsize=14)
    axes[i].set_xlabel(col, fontsize=12)
    axes[i].set_ylabel("Frequency", fontsize=12)

plt.tight_layout()
plt.show()


# In[42]:


# Plot 1: Screen Time vs. Data Usage
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Screen Time (hrs/day)', y='Data Usage (GB/month)', hue='Primary Use', data=df, palette='viridis')
plt.title('Screen Time vs. Data Usage by Primary Use')
plt.xlabel('Screen Time (hrs/day)')
plt.ylabel('Data Usage (GB/month)')
plt.legend(title='Primary Use')
plt.show()


# In[45]:


# Plot 2: Correlation Heatmap
# Selecting only numeric columns for the heatmap
numeric_data = df.select_dtypes(include=['float64', 'int64'])


# In[47]:


# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Mobile Usage Features')
plt.show()


# In[49]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='Primary Use', y='Screen Time (hrs/day)', data=df, hue='Primary Use', palette='Set2', legend=False)
plt.title('Screen Time Distribution by Primary Use')
plt.xlabel('Primary Use')
plt.ylabel('Screen Time (hrs/day)')
plt.xticks(rotation=45)
plt.legend([], [], frameon=False)  # Hide the duplicate legend
plt.show()


# **classification**
# 

# In[52]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score,classification_report 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

# Display the first few rows to inspect the data
print(df.head())

# label encoding for categorical columns
le = LabelEncoder()
columns = ['Gender', 'Location', 'Phone Brand', 'OS', 'Primary Use']

for col in columns:
    df[col]= le.fit_transform(df[col])

# Selecting features for modeling
features = ['Age', 'Gender', 'OS','Screen Time (hrs/day)', 'Data Usage (GB/month)',
                   'Calls Duration (mins/day)', 'Number of Apps Installed',
                   'Social Media Time (hrs/day)', 'E-commerce Spend (INR/month)',
                   'Streaming Time (hrs/day)', 'Gaming Time (hrs/day)',
                   'Monthly Recharge Cost (INR)'
                ]


# In[54]:


df['Primary Use'] = df['Primary Use'].astype(str)  # Convert to string 

X = df[features]

y = df['Primary Use']

y = y.astype(int)



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# In[62]:


import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def clasificaions(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train a model on the training data

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=5, class_weight="balanced", random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        }

    trained_models = {}  # Store trained models
    results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall'])

    # Train the models on the training data
    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=3)  # Reduce CV folds for speed
        avg_cv_score = np.mean(cv_scores)

        # Store trained model
        trained_models[name] = model

        # Append results to DataFrame
        results_df = pd.concat([results_df, pd.DataFrame({
            'Model': [name],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'Cross-Validation Score': [avg_cv_score]
        })], ignore_index=True)
        
        # Print classification report
        print(f"\nDetailed Classification Report for {name}:")
        print(classification_report(y_test, y_pred))

        # Identify the best model based on Accuracy score
        best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
        best_model = trained_models[best_model_name]


        best_model_filename = f"{best_model_name.lower()}_model.pkl"
        with open(best_model_filename, "wb") as model_file:
            pickle.dump(best_model, model_file)

    return results_df, best_model_name, best_model_filename


results_df, best_model_name, best_model_filename = clasificaions(X_scaled, y)
# Print final results
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))

# Print best model
print(f"\n🏆 The best model is: {best_model_name} (Saved as {best_model_filename})")



# **hyper parameter tunning for best model**

# Hyperparameter tuning is used to find the best combination of settings that helps a model perform as accurately and efficiently as possible.

# In[67]:


import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split the data (ensure 'X' and 'y' are already defined)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define hyperparameter grid
param_grid = {
    'max_depth': [5, 10, 15, 20, None],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4, 6]
}

# Initialize Decision Tree
dt = DecisionTreeClassifier(class_weight="balanced", random_state=42)

# Perform GridSearchCV
grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,  # more robust cross-validation
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Best model and parameters
best_dt = grid_search.best_estimator_
print("\n✅ Best Parameters Found:", grid_search.best_params_)

# Evaluate the tuned model
y_pred = best_dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Accuracy on Test Set: {accuracy:.4f}")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Save the tuned model
model_filename = "best_decision_tree_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(best_dt, file)

print(f"\n💾 Tuned Decision Tree model saved as: {model_filename}")


# In[71]:


# Make sure you have your feature names as a list
# Example: if your features came from a DataFrame X, use:
feature_names = X.columns

# Get feature importance from the best Decision Tree model
feature_importance = best_dt.feature_importances_

# Create a DataFrame for better sorting and visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Sort by importance descending
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(12, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color="skyblue")
plt.xlabel("Feature Importance", fontsize=12)
plt.ylabel("Feature Name", fontsize=12)
plt.title("Feature Importance for Optimized Decision Tree", fontsize=14)
plt.gca().invert_yaxis()  # So most important feature is on top
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[73]:


imp_features = ['Age','Screen Time (hrs/day)', 'Data Usage (GB/month)',
                   'Calls Duration (mins/day)', 'Number of Apps Installed',
                   'Social Media Time (hrs/day)', 'E-commerce Spend (INR/month)',
                   'Streaming Time (hrs/day)', 'Gaming Time (hrs/day)',
                   'Monthly Recharge Cost (INR)'
                ]

# Update X dataset
X_filtered = X[imp_features]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)

# Train Decision Tree Again
dt = DecisionTreeClassifier(max_depth= 15, min_samples_leaf= 1, min_samples_split= 10)
dt.fit(X_train, y_train)

# Save the updated model
with open("best_decision_tree_model.pkl", "wb") as model_file:
    pickle.dump(dt, model_file)


# **clustering**

# In[98]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# Assuming df is your original DataFrame with all features
# Step 0: Scale the data
features = df.columns  # or specify the list of feature columns explicitly if needed
X = df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to plot clusters after reducing to 2D using PCA for visualization
def plot_clusters_2d(X_scaled, labels, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.show()

### KMeans ###
distortions = []
silhouette_scores = []
K = range(2, 10)

print("Performing KMeans clustering analysis...")
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    distortions.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

optimal_kmeans_clusters = K[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=optimal_kmeans_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

plot_clusters_2d(X_scaled, kmeans_labels, f'KMeans Clustering (k={optimal_kmeans_clusters})')


scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis')
plt.title(f'Final Clustering (k={optimal_kmeans_clusters})')
plt.colorbar(scatter)

plt.figure(figsize=(12,5))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('KMeans: Elbow Method')
plt.grid()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(K, silhouette_scores, 'ro-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('KMeans: Silhouette Analysis')
plt.grid()
plt.show()

### Hierarchical Clustering ###
n_clusters_range = range(2, 7)
results_hier = {'silhouette_scores': [], 'calinski_scores': []}

print("Performing Hierarchical clustering analysis...")
for n_clusters in tqdm(n_clusters_range, desc="Hierarchical"):
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hierarchical.fit_predict(X_scaled)
    results_hier['silhouette_scores'].append(silhouette_score(X_scaled, labels))
    results_hier['calinski_scores'].append(calinski_harabasz_score(X_scaled, labels))

optimal_hier_clusters = n_clusters_range[np.argmax(results_hier['silhouette_scores'])]
hierarchical = AgglomerativeClustering(n_clusters=optimal_hier_clusters)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

plt.figure(figsize=(8, 5))
plt.plot(list(n_clusters_range), results_hier['silhouette_scores'], marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Hierarchical Clustering: Silhouette Score')
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix, truncate_mode='lastp', p=10)
plt.title('Hierarchical Clustering: Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

plot_clusters_2d(X_scaled, hierarchical_labels, f'Hierarchical Clustering (k={optimal_hier_clusters})')

scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=hierarchical_labels, cmap='viridis')
plt.title(f'Final Clustering (k={optimal_hier_clusters})')
plt.colorbar(scatter)

### DBSCAN ###
print("Performing DBSCAN clustering analysis...")
neighb = NearestNeighbors(n_neighbors=2)
neighb.fit(X_scaled)
distances, _ = neighb.kneighbors(X_scaled)
mean_dist = np.mean(distances[:, 1])
eps_range = np.linspace(mean_dist/2, mean_dist*2, 10)

results_dbscan = {'eps_values': eps_range, 'silhouette_scores': [], 'n_clusters': [], 'n_noise': []}

for eps in tqdm(eps_range, desc="DBSCAN"):
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    if n_clusters >= 2 and n_noise < len(X_scaled):
        valid_points = labels != -1
        if np.sum(valid_points) > 1:
            sil_score = silhouette_score(X_scaled[valid_points], labels[valid_points])
        else:
            sil_score = -1
    else:
        sil_score = -1
    results_dbscan['silhouette_scores'].append(sil_score)
    results_dbscan['n_clusters'].append(n_clusters)
    results_dbscan['n_noise'].append(n_noise)

valid_scores = np.array(results_dbscan['silhouette_scores'])
if np.all(valid_scores == -1):
    optimal_eps = eps_range[np.argmin(results_dbscan['n_noise'])]
else:
    optimal_eps = eps_range[np.argmax(valid_scores)]

dbscan = DBSCAN(eps=optimal_eps, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

plt.figure(figsize=(8, 5))
plt.plot(eps_range, results_dbscan['silhouette_scores'], marker='o')
plt.xlabel('Epsilon')
plt.ylabel('Silhouette Score')
plt.title('DBSCAN: Silhouette Score vs Epsilon')
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(eps_range, results_dbscan['n_clusters'], marker='o')
plt.xlabel('Epsilon')
plt.ylabel('Number of Clusters')
plt.title('DBSCAN: Number of Clusters vs Epsilon')
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(eps_range, results_dbscan['n_noise'], marker='o')
plt.xlabel('Epsilon')
plt.ylabel('Number of Noise Points')
plt.title('DBSCAN: Number of Noise Points vs Epsilon')
plt.grid()
plt.show()

plot_clusters_2d(X_scaled, dbscan_labels, f'DBSCAN Clustering (eps={optimal_eps:.2f})')

scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='viridis')
plt.title(f'Final Clustering (eps={optimal_eps:.2f})')
plt.colorbar(scatter)
plt.show()

print(dbscan_labels, optimal_eps)


### GMM ###
print("Performing Gaussian Mixture Model clustering analysis...")
n_components_range = range(2, 7)
results_gmm = {'silhouette_scores': [], 'bic_scores': [], 'aic_scores': []}

for n_components in tqdm(n_components_range, desc="GMM"):
    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=5)
    labels = gmm.fit_predict(X_scaled)
    results_gmm['silhouette_scores'].append(silhouette_score(X_scaled, labels))
    results_gmm['bic_scores'].append(gmm.bic(X_scaled))
    results_gmm['aic_scores'].append(gmm.aic(X_scaled))

optimal_gmm_components = n_components_range[np.argmax(results_gmm['silhouette_scores'])]
gmm = GaussianMixture(n_components=optimal_gmm_components, random_state=42, n_init=5)
gmm_labels = gmm.fit_predict(X_scaled)

plt.figure(figsize=(8, 5))
plt.plot(list(n_components_range), results_gmm['silhouette_scores'], marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Silhouette Score')
plt.title('GMM: Silhouette Score')
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(list(n_components_range), results_gmm['bic_scores'], marker='o')
plt.xlabel('Number of Components')
plt.ylabel('BIC Score')
plt.title('GMM: BIC Score')
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(list(n_components_range), results_gmm['aic_scores'], marker='o')
plt.xlabel('Number of Components')
plt.ylabel('AIC Score')
plt.title('GMM: AIC Score')
plt.grid()
plt.show()

plot_clusters_2d(X_scaled, gmm_labels, f'GMM Clustering (n={optimal_gmm_components})')
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels, cmap='viridis')
plt.title(f'Final Clustering (n={optimal_gmm_components})')
plt.colorbar(scatter)
plt.show()

### Spectral Clustering ###
print("Performing Spectral Clustering analysis...")
n_clusters_range = range(2, 7)
affinity = 'rbf'
results_spectral = {'silhouette_scores': [], 'inertia': [], 'connectivity': []}

def spectral_pseudo_inertia(X, labels):
    # Sum of squared distances from points to cluster centers (approx)
    centers = np.array([X[labels == i].mean(axis=0) for i in np.unique(labels)])
    inertia = sum(np.sum((X[labels == i] - centers[i]) ** 2) for i in range(len(centers)))
    return inertia

for n_clusters in tqdm(n_clusters_range, desc="Spectral"):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=42)
    labels = spectral.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    inertia = spectral_pseudo_inertia(X_scaled, labels)
    # Connectivity is not defined directly in sklearn spectral, skipping or can implement later
    connectivity = 0  # placeholder
    results_spectral['silhouette_scores'].append(sil_score)
    results_spectral['inertia'].append(inertia)
    results_spectral['connectivity'].append(connectivity)

optimal_spectral_clusters = n_clusters_range[np.argmax(results_spectral['silhouette_scores'])]
spectral = SpectralClustering(n_clusters=optimal_spectral_clusters, affinity=affinity, random_state=42)
spectral_labels = spectral.fit_predict(X_scaled)

plt.figure(figsize=(8, 5))
plt.plot(list(n_clusters_range), results_spectral['silhouette_scores'], marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Spectral Clustering: Silhouette Score')
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(list(n_clusters_range), results_spectral['inertia'], marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Pseudo-Inertia')
plt.title('Spectral Clustering: Pseudo Inertia')
plt.grid()
plt.show()

plot_clusters_2d(X_scaled, spectral_labels, f'Spectral Clustering (k={optimal_spectral_clusters})')
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=spectral_labels, cmap='viridis')
plt.title(f'Final Clustering (k={optimal_spectral_clusters})')
plt.colorbar(scatter)
plt.show()
### Save Results ###
df['KMeans_Labels'] = kmeans_labels
df['Hierarchical_Labels'] = hierarchical_labels
df['DBSCAN_Labels'] = dbscan_labels
df['GMM_Labels'] = gmm_labels
df['Spectral_Labels'] = spectral_labels

df.to_csv('clustering.csv', index=False)

print("Clustering completed and results saved to 'clustering_results.csv'.")


# In[96]:


print(df.columns)


# In[99]:


print(df.columns.tolist())


# In[ ]:




