import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# STEP 1: Load Dataset
df = pd.read_csv(
    r"C:\Users\Aashish\OneDrive\Documents\Downloads\daily-renewable-energy-generation.csv"
)

print("Dataset Shape:", df.shape)
print(df.head())
print("\nMissing Values:\n", df.isnull().sum())

# STEP 2: Data Preparation
# Dataset date format is YYYY-MM-DD
df["date"] = pd.to_datetime(
    df["date"], format="%Y-%m-%d", errors="coerce"
)

df["region"] = df["region"].fillna("Unknown")

# Remove rows with invalid date or missing target
df = df.dropna(subset=["date", "solar_energy"]).copy()

# Remove All India aggregate
df = df[df["state_name"] != "All India"].copy()

# Sort data by date
df = df.sort_values("date").reset_index(drop=True)

print("\nRows after cleaning:", len(df))
print("Unique states:", df["state_name"].nunique())

# Check duplicate date-state records
print(
    "\nDuplicate Date-State Records:",
    df.duplicated(subset=["date", "state_name"]).sum()
)

# STEP 3: Exploratory Data Analysis (EDA)

# 3.1 Solar Energy Distribution
df["solar_energy"].hist(bins=30)
plt.title("Solar Energy Distribution")
plt.xlabel("Solar Energy")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 3.2 Top 10 States by Average Solar Energy
state_solar = (
    df.groupby("state_name")["solar_energy"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

if not state_solar.empty:
    state_solar.plot(kind="bar")
    plt.title("Top 10 States by Average Solar Energy")
    plt.xlabel("State")
    plt.ylabel("Average Solar Energy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No state data available for the graph.")

# 3.3 Correlation Heatmap
energy_cols = [
    "wind_energy",
    "solar_energy",
    "other_renewable_energy",
    "total_renewable_energy"
]

sns.heatmap(
    df[energy_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Energy Correlation")
plt.tight_layout()
plt.show()

# 3.4 Daily Solar Energy Trend
daily_solar = df.groupby("date")["solar_energy"].sum()

daily_solar.plot(figsize=(10, 4))
plt.title("Daily Solar Energy Generation")
plt.xlabel("Date")
plt.ylabel("Solar Energy")
plt.tight_layout()
plt.show()

# STEP 4: Feature Engineering
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek

# STEP 5: Select Features and Target
features = ["month", "day_of_week"]

X = df[features]
y = df["solar_energy"]

# STEP 6: Train-Test Split (preserve time order)
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("\nTraining Rows:", len(X_train))
print("Testing Rows:", len(X_test))

# STEP 7: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# STEP 8: Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("R-squared:", round(r2, 3))

# STEP 9: Actual vs Predicted Graph
plt.figure(figsize=(10, 4))
plt.plot(y_test.to_numpy()[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")

plt.title("Actual vs Predicted Solar Energy")
plt.xlabel("Test Record")
plt.ylabel("Solar Energy")
plt.legend()
plt.tight_layout()
plt.show()
