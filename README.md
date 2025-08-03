# Decoding-phone-usage-pattern-in-india
Analyzed Indian smartphone user behavior using ML and clustering on usage data. Predicted primary device use (Education, Gaming, etc.) and identified user segments. Built a Streamlit app for EDA, classification, and clustering insights. Skills: Python, EDA, ML, Streamlit, User Behavior Analysis.          

##  Project Overview
This project aims to analyze mobile usage behavior in India using user demographic and device usage data. The goal is to classify users based on their primary mobile use (e.g., Education, Gaming, Social Media, Entertainment) and identify distinct usage patterns through clustering.

##  Problem Statement
Design a system to:
- Analyze mobile usage behavior
- Classify users' primary mobile use
- Cluster users based on device usage patterns
- Build an interactive app to visualize insights

## Skills & Technologies
- Python Scripting
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Machine Learning (Multiclass Classification)
- Clustering Algorithms
- Streamlit (for UI)
- Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

##  Dataset
**Filename:** `phone_usage_india.csv`  
**Features:**
- User demographics (Age, Gender, Location)
- Device specs (Phone Brand, OS)
- Usage stats (Screen Time, Data Usage, Call Duration, App Count)
- Activity time (Social Media, Streaming, Gaming)
- Monthly recharge and e-commerce spending
- Target: `Primary Use`

##  Key Analyses
- Correlation between screen time and battery usage
- Data and app usage by age groups
- Clustering of users based on behavior
- Classification of primary use with ML models

##  Machine Learning Models
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting (XGBoost/LightGBM)

### Clustering Techniques:
- K-Means
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Models

##  Evaluation Metrics
- Accuracy, Precision, Recall (Classification)
- Silhouette Score (Clustering)

##  Streamlit Application
An interactive web app showcasing:
- EDA Visualizations
- Primary Use Prediction
- User Clustering Insights

##  How to Run
```bash
streamlit run app.py
