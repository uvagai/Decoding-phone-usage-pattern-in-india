import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(
    page_title="Phone Usage Analysis Dashboard",
    page_icon="📱",
    layout="wide"
)

USAGE_CATEGORIES = {
    0: "Social Media",
    1: "Gaming",
    2: "Streaming",
    3: "Professional",
    4: "Educational",
    5: "Communication"
}

# Sidebar Navigation
page = st.sidebar.selectbox(
    "📁 Select a Section",
    ["Dashboard Overview", "Usage Patterns", "Classification", "User Segmentation"]
)

# Load the dataset
df = pd.read_csv("clustering.csv")

# Load the Decision Tree model
try:
    with open("decision tree_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    model = None

# ---------------- Dashboard Overview ---------------- #
if page == "Dashboard Overview":
    st.header("📊 Dashboard Overview")
    st.write("This is a Phone Usage Analysis application that allows users to analyze and predict phone usage patterns.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Users", len(df))
        st.metric("Average Screen Time", f"{df['Screen Time (hrs/day)'].mean():.1f} hrs/day")

    with col2:
        st.metric("Average Data Usage", f"{df['Data Usage (GB/month)'].mean():.1f} GB/month")

        df['Primary Use'] = df['Primary Use'].fillna(-1).astype(int)
        try:
            use_counts = df['Primary Use'].value_counts()
            most_common = USAGE_CATEGORIES.get(int(use_counts.idxmax()), "Unknown Category")
        except Exception as e:
            st.error(f"Error calculating most common use: {str(e)}")
            most_common = "Error in calculation"

        st.metric("Most Common Use", most_common)

# ---------------- Usage Patterns ---------------- #
elif page == "Usage Patterns":
    st.header("📈 Usage Patterns")
    st.subheader("📊 Select Feature for Visualization")

    feature_options = [
        "Screen Time (hrs/day)", "Social Media Time (hrs/day)", 
        "Streaming Time (hrs/day)", "Gaming Time (hrs/day)", 
        "Data Usage (GB/month)", "E-commerce Spend (INR/month)", 
        "Monthly Recharge Cost (INR)"
    ]

    selected_feature = st.selectbox("Choose a feature to visualize", feature_options)

    st.subheader(f"📊 Distribution of {selected_feature}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[selected_feature], bins=30, kde=True, ax=ax)
    plt.xlabel(selected_feature)
    plt.ylabel("Count")
    plt.title(f"Distribution of {selected_feature}")
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("📈 Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    correlation = df[numeric_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    plt.title("Correlation between Features")
    st.pyplot(fig2)

# ---------------- Classification ---------------- #
elif page == "Classification":
    st.header("🎯 User Classification")

    if model is not None:
        st.subheader("Predict Primary Use")
        with st.form("classification_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Age", 18, 80, 25)
                screen_time = st.number_input("Screen Time (hrs/day)", 0.5, 24.0, 4.0, 0.5)
                data_usage = st.number_input("Data Usage (GB/month)", 1.0, 100.0, 30.0, 1.0)
                calls_duration = st.number_input("Calls Duration (mins/day)", 1.0, 300.0, 45.0, 5.0)

            with col2:
                num_apps = st.number_input("Number of Apps Installed", 5, 200, 25, 1)
                social_media = st.number_input("Social Media Time (hrs/day)", 0.0, 24.0, 2.5, 0.5)
                ecommerce_spend = st.number_input("E-commerce Spend (INR/month)", 0.0, 10000.0, 2000.0, 100.0)
                streaming_time = st.number_input("Streaming Time (hrs/day)", 0.0, 24.0, 1.5, 0.5)

            with col3:
                gaming_time = st.number_input("Gaming Time (hrs/day)", 0.0, 24.0, 1.0, 0.5)
                monthly_cost = st.number_input("Monthly Recharge Cost (INR)", 200.0, 5000.0, 699.0, 50.0)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                os = st.selectbox("Operating System", ["Android", "iOS"])

            submit_button = st.form_submit_button("Predict Primary Use")

        if submit_button:
            total_time = screen_time + social_media + gaming_time + streaming_time
            if total_time > 24:
                st.error("❌ Total time across activities exceeds 24 hours. Please adjust your inputs.")
            else:
                try:
                    gender_mapping = {"Male": 0, "Female": 1, "Other": 2}
                    os_mapping = {"Android": 0, "iOS": 1}
                    input_data = pd.DataFrame([[
                        age, gender_mapping[gender], screen_time, data_usage, calls_duration, num_apps,
                        social_media, ecommerce_spend, streaming_time, gaming_time, monthly_cost, os_mapping[os]
                    ]], columns=[
                        'Age', 'Gender', 'Screen Time (hrs/day)', 'Data Usage (GB/month)',
                        'Calls Duration (mins/day)', 'Number of Apps Installed',
                        'Social Media Time (hrs/day)', 'E-commerce Spend (INR/month)',
                        'Streaming Time (hrs/day)', 'Gaming Time (hrs/day)',
                        'Monthly Recharge Cost (INR)', 'OS'
                    ])
                    prediction_num = model.predict(input_data)[0]
                    prediction_category = USAGE_CATEGORIES.get(prediction_num, "Unknown Category")
                    st.success(f"🎯 Predicted Primary Use: {prediction_category}")
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")

# ---------------- User Segmentation ---------------- #
elif page == "User Segmentation":
    st.header("👥 User Segmentation")

    cluster_columns = [col for col in df.columns if col.startswith('cluster_')]

    if cluster_columns:
        st.subheader("User Clusters")
        cluster_col = st.selectbox("Select Cluster Type", cluster_columns)

        def create_cluster_plot(df, cluster_col):
            plt.style.use('seaborn-v0_8')
            fig, ax = plt.subplots(figsize=(10, 6))
            n_clusters = len(df[cluster_col].unique())
            palette = sns.color_palette("husl", n_clusters)
            sns.scatterplot(data=df,
                            x='Screen Time (hrs/day)',
                            y='Data Usage (GB/month)',
                            hue=cluster_col,
                            palette=palette,
                            alpha=0.6)
            plt.title('User Segments by Screen Time and Data Usage')
            plt.xlabel('Screen Time (hrs/day)')
            plt.ylabel('Data Usage (GB/month)')
            plt.grid(True, linestyle='--', alpha=0.7)
            return fig

        fig = create_cluster_plot(df, cluster_col)
        st.pyplot(fig)
        plt.close()

        st.subheader("Cluster Characteristics")
        for cluster in sorted(df[cluster_col].unique()):
            cluster_data = df[df[cluster_col] == cluster]
            with st.expander(f"Cluster {cluster}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Number of Users", len(cluster_data))
                    st.metric("Avg Screen Time", f"{cluster_data['Screen Time (hrs/day)'].mean():.1f} hrs/day")
                with col2:
                    st.metric("Avg Data Usage", f"{cluster_data['Data Usage (GB/month)'].mean():.1f} GB/month")
                most_common_use = USAGE_CATEGORIES.get(
                    cluster_data['Primary Use'].astype(int).mode()[0],
                    f"Category {cluster_data['Primary Use'].mode()[0]}"
                )
                st.metric("Most Common Use", most_common_use)
    else:
        st.warning("⚠️ No cluster columns found in the dataset.")

# ---------------- Footer ---------------- #
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
    <p>made by uvi • Decoding Phone Usage Patterns (Analysis and Visualization)</p>
    </div>
""", unsafe_allow_html=True)
