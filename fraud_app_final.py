# fraud_app_final.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------
# Load trained model
# ---------------------------
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Load dataset for samples and charts
# ---------------------------
df = pd.read_csv(r"C:\Users\balaj\Documents\DataProjects\Fraud-Detection-Project\Data\fraudTest.csv")

# Ensure required datetime columns exist
if "hour" not in df.columns:
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["weekday"] = df["trans_date_trans_time"].dt.weekday
    df["month"] = df["trans_date_trans_time"].dt.month

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    page_icon="💳",
)

# ---------------------------
# CSS: Dark mode
# ---------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #E0E0E0;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #333333;
    }
    .stSidebar {
        background-color: #1F1F1F;
    }
    .stDataFrame div.row_widget {
        background-color: #1E1E1E;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Transaction Details")

# Transaction Info
st.sidebar.subheader("Transaction Info")
amount = st.sidebar.slider("Transaction Amount (USD)", 0, 5000, 1000)
hour = st.sidebar.slider("Hour of Transaction", 0, 23, 12)
day = st.sidebar.slider("Day of Transaction", 1, 31, 15)
weekday = st.sidebar.selectbox("Weekday (0=Mon, 6=Sun)", list(range(7)))
month = st.sidebar.selectbox("Month", list(range(1, 13)))

# User Info
st.sidebar.subheader("User Info")
gender = st.sidebar.selectbox("Gender", ["M", "F"])
lat = st.sidebar.number_input("User Latitude", value=12.97)
long = st.sidebar.number_input("User Longitude", value=77.59)
city_pop = st.sidebar.number_input("City Population", value=1000000)

# Transaction Category & City
st.sidebar.subheader("Merchant Info")
category = st.sidebar.selectbox("Transaction Category", df["category"].unique())
city_name = st.sidebar.selectbox("City Name", df["city"].unique())

# Optional: button
predict_btn = st.sidebar.button("Predict Fraud")

# ---------------------------
# Prepare input for model
# ---------------------------
if predict_btn:
    sample = pd.DataFrame({
        "amt": [amount],
        "hour": [hour],
        "day": [day],
        "weekday": [weekday],
        "month": [month],
        "gender": [gender],
        "lat": [lat],
        "long": [long],
        "city_pop": [city_pop],
        "category": [category],
        "city": [city_name]
    })

    # ---------------------------
    # Prediction
    # ---------------------------
    prediction = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]  # probability of fraud

    # ---------------------------
    # Layout: Two columns
    # ---------------------------
    col1, col2 = st.columns([1, 1])

    # Left Column: Prediction + Probability
    with col1:
        st.markdown("### Prediction Result")
        if prediction == 1:
            st.markdown("❌ **Fraud**", unsafe_allow_html=True)
        else:
            st.markdown("✅ **Not Fraud**", unsafe_allow_html=True)

        # Probability Donut Chart
        fig = go.Figure(go.Pie(
            values=[1 - prob, prob],
            labels=["Safe", "Fraud"],
            hole=0.6,
            marker_colors=["#28A745", "#DC3545"]
        ))
        fig.update_layout(showlegend=True, paper_bgcolor="#121212", font_color="#E0E0E0")
        st.plotly_chart(fig, use_container_width=True)

    # Right Column: Feature Importance + Data preview
    with col2:
        st.markdown("### Feature Importance (Top 5)")
        if hasattr(model.named_steps["model"], "feature_importances_"):
            importances = model.named_steps["model"].feature_importances_
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
            feat_imp = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(10)
            fig2 = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
                          color="Importance", color_continuous_scale="Blues")
            fig2.update_layout(paper_bgcolor="#121212", plot_bgcolor="#121212", font_color="#E0E0E0")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Sample Data Preview")
        sample_data = df.sample(5)
        st.dataframe(sample_data.style.applymap(
            lambda x: "background-color: red" if x==1 else "",
            subset=["is_fraud"]
        ))

    # ---------------------------
    # Optional Charts: Hour-wise Fraud Trend
    # ---------------------------
    st.markdown("### Hour-wise Fraud Trend (Sample Data)")
    if "hour" in df.columns:
        hour_df = df.groupby("hour")["is_fraud"].mean().reset_index()
        fig3 = px.line(hour_df, x="hour", y="is_fraud", markers=True,
                       labels={"hour":"Hour of Day", "is_fraud":"Fraud Probability"})
        fig3.update_layout(paper_bgcolor="#121212", plot_bgcolor="#121212", font_color="#E0E0E0")
        st.plotly_chart(fig3, use_container_width=True)

    # ---------------------------
    # Optional: Footer
    # ---------------------------
    st.markdown("---")
    st.markdown("Made by **Balaji R H** | [GitHub](https://github.com/ROCKYBH7)", unsafe_allow_html=True)
