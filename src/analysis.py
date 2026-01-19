import matplotlib.pyplot as plt
from src.data_processing import load_and_clean_data


df = load_and_clean_data("data/sample_sales.csv")

print("Basic statistics:")
print(df["sales"].describe())

plt.figure()
plt.plot(df["date"], df["sales"])
plt.title("Daily Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()