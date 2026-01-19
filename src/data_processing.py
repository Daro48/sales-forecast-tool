import pandas as pd

def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Load sales data from CSV and Perform basic cleaning.
    """
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    df["sales"] = df["sales"].ffill()

    return df

if __name__ == "__main__":
    df = load_and_clean_data("data/sample_sales.csv")
    print(df.head())