import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.data_processing import load_and_clean_data


def train_sales_model(df: pd.DataFrame):
    """
    Train a simple regression model to predict sales.
    """

    # Feature: day index
    df["day_index"] = range(len(df))

    X = df[["day_index"]]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    return model, score


def predict_future_sales(model, last_day_index: int, days_ahead: int = 7):
    """
    Predict future sales for the next N days.
    """
    future_days = pd.DataFrame({
    "day_index": [last_day_index + i for i in range(1, days_ahead + 1)]
})

    predictions = model.predict(future_days)

    return predictions


if __name__ == "__main__":
    df = load_and_clean_data("data/sample_sales.csv")

    model, score = train_sales_model(df)
    print(f"Model R² score: {score:.2f}")

    predictions = predict_future_sales(
        model,
        last_day_index=len(df) - 1,
        days_ahead=7
    )

    print("Next 7 days prediction:")
    for i, value in enumerate(predictions, start=1):
        print(f"Day {i}: {value:.2f}")

