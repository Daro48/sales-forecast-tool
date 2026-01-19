import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/sample_sales.csv")

df["date"] = pd.to_datetime(df["date"])

print("Basic statistics: ")
print(df["sales"].describe())

df = df.sort_values("date")

plt.figure()
plt.plot(df["date"], df["sales"])
plt.title("Daily Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()
