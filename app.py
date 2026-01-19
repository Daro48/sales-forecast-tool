import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_processing import load_and_clean_data
from src.model import train_linear_model, train_random_forest_model, predict_future_sales

st.set_page_config(page_title="Sales Forecast Tool", layout="centered")

st.title("📈 Sales Forecast Tool")
st.write("Upload sales data and predict future sales using Machine Learning.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df["date"] = pd.to_datetime(df["date"])

    st.subheader("📊 Historical Sales Data")
    st.dataframe(df)

    lin_model, lin_score = train_linear_model(df)
    rf_model, rf_score = train_random_forest_model(df)

    st.subheader("📐 Model Comparison")
    st.write(f"Linear Regression R²: **{lin_score:.2f}**")
    st.write(f"Random Forest R²: **{rf_score:.2f}**")

    st.info(
        "Linear Regression is used for forecasting because it extrapolates trends "
        "more reliably in time series data. Random Forest is shown for comparison."
    )

    model = lin_model

    days_ahead = st.slider("Days to predict", 1, 30, 7)

    predictions = predict_future_sales(
        model,
        last_day_index=len(df) - 1,
        days_ahead=days_ahead
    )

    st.subheader("🔮 Sales Forecast")

    fig, ax = plt.subplots()
    ax.plot(df["date"], df["sales"], label="Historical Sales")

    future_dates = pd.date_range(
        start=df["date"].iloc[-1] + pd.Timedelta(days=1),
        periods=len(predictions),
        freq="D"
    )

    ax.plot(future_dates, predictions, linestyle="--", label="Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig)
