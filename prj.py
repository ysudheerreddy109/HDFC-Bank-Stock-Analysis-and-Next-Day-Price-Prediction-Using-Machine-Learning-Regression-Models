
# ============================================================
# PROJECT: HDFC BANK STOCK ANALYSIS AND MACHINE LEARNING
# Dataset: HDFCBANK.NS.csv
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

FILE_PATH = "HDFCBANK.NS.csv"
OUTPUT_DIR = "HDFC_Project_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(FILE_PATH)

print("=" * 60)
print("HDFC BANK DATA SCIENCE AND MACHINE LEARNING PROJECT")
print("=" * 60)

print("\nFirst five records:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 2. DATA PREPROCESSING AND CLEANING
# ------------------------------------------------------------

print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

df["Date"] = pd.to_datetime(
    df["Date"], dayfirst=True, errors="coerce"
)

numeric_cols = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Sort dates and remove duplicates
df = df.sort_values("Date")
df = df.drop_duplicates().reset_index(drop=True)

# Handle invalid values
df.loc[df["Volume"] < 0, "Volume"] = np.nan

invalid_range = df["High"] < df["Low"]
df.loc[invalid_range, ["High", "Low"]] = np.nan

# Remove records without essential information
df = df.dropna(
    subset=["Date", "Close", "Adj Close"]
).copy()

# Fill remaining gaps in input columns
for col in ["Open", "High", "Low", "Volume"]:
    df[col] = df[col].interpolate(
        limit_direction="both"
    )

df["Volume"] = df["Volume"].fillna(0)

df = df.sort_values("Date").reset_index(drop=True)

print("\nDataset shape after cleaning:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

df.to_csv(
    os.path.join(OUTPUT_DIR, "HDFCBANK_cleaned.csv"),
    index=False
)

# ------------------------------------------------------------
# 3. OBJECTIVE 1: ANALYZE STOCK PRICE TRENDS
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["Date"],
    df["Adj Close"],
    linewidth=1
)

plt.title("HDFC Bank Adjusted Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "01_price_trend.png"),
    dpi=150
)
plt.show()

print("\nObjective 1:")
print("Minimum adjusted close:", df["Adj Close"].min())
print("Maximum adjusted close:", df["Adj Close"].max())

# ------------------------------------------------------------
# 4. OBJECTIVE 2: ANALYZE TRADING VOLUME
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    df["Date"],
    df["Volume"],
    linewidth=0.8
)

plt.title("HDFC Bank Daily Trading Volume")
plt.xlabel("Date")
plt.ylabel("Trading Volume")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "02_trading_volume.png"),
    dpi=150
)
plt.show()

print("\nObjective 2:")
print("Average daily volume:", round(df["Volume"].mean(), 2))

# ------------------------------------------------------------
# 5. OBJECTIVE 3: DAILY RETURNS AND VOLATILITY
# ------------------------------------------------------------

df["Daily_Return"] = df["Adj Close"].pct_change()

returns = df["Daily_Return"].dropna()

print("\nObjective 3:")
print("Mean daily return:", round(returns.mean(), 6))
print("Daily volatility:", round(returns.std(), 6))

lower = returns.quantile(0.01)
upper = returns.quantile(0.99)

plt.figure(figsize=(10, 5))

plt.hist(
    returns.clip(lower, upper),
    bins=50,
    edgecolor="black"
)

plt.title("Distribution of Daily Stock Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "03_return_distribution.png"),
    dpi=150
)
plt.show()

# ------------------------------------------------------------
# 6. OBJECTIVE 4: CORRELATION ANALYSIS
# ------------------------------------------------------------

corr = df[numeric_cols].corr()

plt.figure(figsize=(9, 6))

plt.imshow(
    corr,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=35,
    ha="right"
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(
            j, i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.title("Correlation Matrix of Market Variables")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "04_correlation_matrix.png"),
    dpi=150
)
plt.show()

print("\nObjective 4: Correlation matrix")
print(corr.round(3))

# ------------------------------------------------------------
# 7. OBJECTIVE 5: YEAR-END PRICE ANALYSIS
# ------------------------------------------------------------

year_end = (
    df.set_index("Date")["Adj Close"]
    .resample("YE")
    .last()
    .dropna()
)

plt.figure(figsize=(10, 5))

plt.plot(
    year_end.index,
    year_end.values,
    marker="o"
)

plt.title("HDFC Bank Year-End Adjusted Closing Price")
plt.xlabel("Year")
plt.ylabel("Adjusted Close Price")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "05_year_end_prices.png"),
    dpi=150
)
plt.show()

print("\nObjective 5: Year-end prices")
print(year_end)

# ------------------------------------------------------------
# 8. FEATURE ENGINEERING
# ------------------------------------------------------------

# Calculate historical returns
df["Return_1d"] = df["Adj Close"].pct_change()
df["Return_5d"] = df["Adj Close"].pct_change(5)

# Moving averages
df["MA_5"] = df["Adj Close"].rolling(5).mean()
df["MA_10"] = df["Adj Close"].rolling(10).mean()
df["MA_20"] = df["Adj Close"].rolling(20).mean()

# Rolling volatility
df["Volatility_10d"] = (
    df["Return_1d"].rolling(10).std()
)

# Daily price range
df["Range_pct"] = (
    (df["High"] - df["Low"])
    / df["Close"].replace(0, np.nan)
)

# Calendar features
df["Day_of_week"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

# Lag features
for lag in [1, 2, 3, 5]:
    df[f"AdjClose_lag_{lag}"] = (
        df["Adj Close"].shift(lag)
    )

# Target: next trading row's adjusted close
df["Target"] = df["Adj Close"].shift(-1)

features = [
    "Open", "High", "Low", "Close", "Volume",
    "Return_1d", "Return_5d",
    "MA_5", "MA_10", "MA_20",
    "Volatility_10d", "Range_pct",
    "Day_of_week", "Month", "Year",
    "AdjClose_lag_1", "AdjClose_lag_2",
    "AdjClose_lag_3", "AdjClose_lag_5"
]

data = df.dropna(
    subset=features + ["Target"]
).copy()

X = data[features].replace(
    [np.inf, -np.inf], np.nan
)

y = data["Target"]

# Chronological split: first 80% for training,
# final 20% for testing
split = int(len(data) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ------------------------------------------------------------
# 9. BUILD FIVE MACHINE LEARNING MODELS
# ------------------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=1.0),

    "Random Forest": RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

results = []
predictions = {}
trained_models = {}

# Train and evaluate each model
for name, model in models.items():

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    trained_models[name] = pipeline
    predictions[name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

# ------------------------------------------------------------
# 10. COMPARE MODEL PERFORMANCE
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "RMSE"
).reset_index(drop=True)

print("\n" + "=" * 60)
print("MACHINE LEARNING MODEL RESULTS")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_results.csv"),
    index=False
)

# Select the model with the lowest test RMSE
best_name = results_df.loc[0, "Model"]
best_pred = predictions[best_name]

print("\nBest model by RMSE:", best_name)

print("MAE:", round(
    results_df.loc[0, "MAE"], 4
))

print("RMSE:", round(
    results_df.loc[0, "RMSE"], 4
))

print("R2:", round(
    results_df.loc[0, "R2"], 4
))

# ------------------------------------------------------------
# 11. VISUALIZE MODEL COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.bar(
    results_df["Model"],
    results_df["RMSE"]
)

plt.title("Machine Learning Model Comparison")
plt.xlabel("Model")
plt.ylabel("RMSE (lower is better)")
plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "06_model_comparison.png"),
    dpi=150
)
plt.show()

# ------------------------------------------------------------
# 12. ACTUAL VS PREDICTED PRICES
# ------------------------------------------------------------

test_dates = data["Date"].iloc[split:]

plt.figure(figsize=(12, 5))

plt.plot(
    test_dates,
    y_test.values,
    label="Actual Price",
    linewidth=1
)

plt.plot(
    test_dates,
    best_pred,
    label=f"Predicted: {best_name}",
    linewidth=1
)

plt.title("Actual vs Predicted Next-Day Adjusted Close")
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "07_actual_vs_predicted.png"),
    dpi=150
)
plt.show()

# ------------------------------------------------------------
# 13. FINAL FINDINGS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print("Date range:",
      df["Date"].min().date(),
      "to",
      df["Date"].max().date())

print("Cleaned dataset rows:", len(df))
print("Modeling dataset rows:", len(data))
print("Best model:", best_name)

print("\nFive project objectives completed:")
print("1. Historical stock price trend analysis")
print("2. Trading volume analysis")
print("3. Daily returns and volatility analysis")
print("4. Correlation analysis")
print("5. Year-end price analysis")

print("\nMachine learning completed:")
print("1. Linear Regression")
print("2. Ridge Regression")
print("3. Random Forest")
print("4. Gradient Boosting")
print("5. Extra Trees")

print("\nAll charts and model results are saved in:")
print(os.path.abspath(OUTPUT_DIR))

print("\nProject execution completed successfully!")
print("Note: Educational analysis only, not investment advice.")
