import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.data_processing import load_and_clean_data


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    df["day_index"] = range(len(df))
    X = df[["day_index"]]
    y = df["sales"]
    return X, y


def train_linear_model(df: pd.DataFrame):
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    return model, score


def train_random_forest_model(df: pd.DataFrame):
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    return model, score


def predict_future_sales(model, last_day_index: int, days_ahead: int = 7):
    future_days = pd.DataFrame({
        "day_index": [last_day_index + i for i in range(1, days_ahead + 1)]
    })
    return model.predict(future_days)


if __name__ == "__main__":
    df = load_and_clean_data("data/sample_sales.csv")

    lin_model, lin_score = train_linear_model(df)
    rf_model, rf_score = train_random_forest_model(df)

    print(f"Linear Regression R²: {lin_score:.2f}")
    print(f"Random Forest R²:     {rf_score:.2f}")

    predictions = predict_future_sales(
        rf_model,
        last_day_index=len(df) - 1,
        days_ahead=7
    )

    print("Random Forest Forecast:")
    for i, value in enumerate(predictions, start=1):
        print(f"Day +{i}: {value:.2f}")
