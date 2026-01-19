import matplotlib.pyplot as plt
import pandas as pd

from src.data_processing import load_and_clean_data
from model import train_sales_model, predict_future_sales


def plot_sales_and_forecast(df, predictions):
    plt.figure()
    plt.plot(df["date"], df["sales"], label="Historical Sales")

    last_date = df["date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(predictions),
        freq="D"
    )

    plt.plot(future_dates, predictions, label="Forecast", linestyle="--")

    plt.title("Sales Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    df = load_and_clean_data("data/sample_sales.csv")

    model, score = train_sales_model(df)
    predictions = predict_future_sales(
        model,
        last_day_index=len(df) - 1,
        days_ahead=7
    )

    plot_sales_and_forecast(df, predictions)
