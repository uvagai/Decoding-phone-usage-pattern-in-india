import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Page config
st.set_page_config(page_title="Phone Usage Dashboard", layout="wide")

# Load dataset (CSV version)
df = pd.read_csv("clustering.csv")  # Ensure this file exists in the directory

# Detect cluster label columns
cluster_columns = [col for col in df.columns if 'label' in col.lower() or 'cluster' in col.lower()]

if not cluster_columns:
    st.warning("⚠️ No cluster columns found in the dataset.")
    st.stop()

# Sidebar
st.sidebar.title("Navigation")
selected_cluster = st.sidebar.selectbox("Select clustering method", cluster_columns)
show_summary = st.sidebar.checkbox("Show Summary Statistics", True)

# Main dashboard title
st.title("📱 Decoding Phone Usage Dashboard")
st.markdown("Explore insights on user behavior, app engagement, and spending patterns.")

# Summary stats
if show_summary:
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Screen Time", f"{df['Screen Time (hrs/day)'].mean():.1f} hrs/day")
    col2.metric("Avg Data Usage", f"{df['Data Usage (GB/month)'].mean():.1f} GB/month")
    col3.metric("Avg Recharge", f"₹{df['Monthly Recharge Cost (INR)'].mean():.0f}/month")

# Add selected cluster to a unified column
df["Selected Cluster"] = df[selected_cluster].astype(str)

# Cluster counts
st.subheader(f"Cluster Counts using {selected_cluster}")
cluster_counts = df["Selected Cluster"].value_counts().sort_index()
st.bar_chart(cluster_counts)

# Feature-wise averages per cluster
st.subheader("📊 Feature Comparison by Cluster")
features = [
    'Screen Time (hrs/day)', 'Data Usage (GB/month)',
    'Calls Duration (mins/day)', 'Number of Apps Installed',
    'Social Media Time (hrs/day)', 'Streaming Time (hrs/day)',
    'Gaming Time (hrs/day)', 'E-commerce Spend (INR/month)',
    'Monthly Recharge Cost (INR)'
]

mean_values = df.groupby("Selected Cluster")[features].mean()
st.dataframe(mean_values.style.highlight_max(axis=0), use_container_width=True)

# Correlation Heatmap
st.subheader("📈 Correlation Heatmap")
corr = df[features].corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig)

# Optional: Cluster-wise visualization with Plotly
st.subheader("🌀 2D Cluster Scatter Plot")
x_axis = st.selectbox("X-axis", features, index=0)
y_axis = st.selectbox("Y-axis", features, index=1)

fig2 = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color="Selected Cluster",
    title=f"{x_axis} vs {y_axis} by Cluster",
    hover_data=["Age", "Gender", "Primary Use"]
)
st.plotly_chart(fig2, use_container_width=True)
