"""
Week 1: Strategic Planning and Data Exploration in Logistics
Author: Sujal Pradeep More
Scenario: Demand-driven inventory and last-mile delivery optimization
for an e-commerce distributor (synthetic data used for illustration).
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# 1. Simulated data (replace with real CSVs, e.g. DataCo / Olist)
# ---------------------------------------------------------------
n = 5000
order_date = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 270, n), unit="D")
distance = rng.gamma(4, 25, n).round(1)
weight = rng.gamma(2, 3, n).round(2)
lead_hrs = 12 + distance * 0.25 + rng.normal(0, 6, n)
df = pd.DataFrame({
    "order_id": np.arange(n),
    "warehouse_id": rng.choice(["Mumbai", "Delhi", "Bengaluru"], n),
    "zone": rng.choice(["A", "B", "C", "D"], n),
    "order_date": order_date,
    "distance_km": distance,
    "weight_kg": weight,
    "transport_cost": (30 + distance * 1.8 + weight * 4 + rng.normal(0, 10, n)).round(2),
    "stock_available": rng.choice([0, 1], n, p=[0.07, 0.93]),
})
df["delivery_date"] = df["order_date"] + pd.to_timedelta(lead_hrs, unit="h")
df["promised_date"] = df["order_date"] + pd.to_timedelta(48, unit="h")
df.loc[rng.choice(n, 150, replace=False), "distance_km"] = np.nan  # missing values

# ---------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------
df = df.drop_duplicates(subset="order_id")
df["distance_km"] = df["distance_km"].fillna(df.groupby("zone")["distance_km"].transform("median"))

# ---------------------------------------------------------------
# 3. KPI calculation
# ---------------------------------------------------------------
df["lead_time_hrs"] = (df["delivery_date"] - df["order_date"]).dt.total_seconds() / 3600
df["on_time"] = (df["delivery_date"] <= df["promised_date"]).astype(int)

otd_rate = df["on_time"].mean() * 100
avg_lead_time = df["lead_time_hrs"].mean()
cost_per_delivery = df["transport_cost"].sum() / len(df)
stockout_rate = (df["stock_available"] == 0).mean() * 100

print(f"On-Time Delivery Rate : {otd_rate:.1f}%")
print(f"Avg Lead Time         : {avg_lead_time:.1f} hrs")
print(f"Cost per Delivery     : {cost_per_delivery:.2f}")
print(f"Stock-Out Rate        : {stockout_rate:.1f}%")

# ---------------------------------------------------------------
# 4. Exploratory analysis
# ---------------------------------------------------------------
print(df[["lead_time_hrs", "distance_km", "transport_cost"]].describe())
sns.heatmap(df[["lead_time_hrs", "distance_km", "weight_kg", "transport_cost"]].corr(),
            annot=True, cmap="Blues")
plt.title("Correlation of Key Logistics Variables")
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 5. Demand forecasting model (weekly units sold)
# ---------------------------------------------------------------
weeks = pd.date_range("2025-01-05", periods=90, freq="W")
units = 200 + 30 * np.sin(np.arange(90) / 8) + rng.normal(0, 10, 90)
weekly = pd.DataFrame({"week": weeks, "units_sold": units})
weekly["lag_1"] = weekly["units_sold"].shift(1)
weekly["lag_4"] = weekly["units_sold"].shift(4)
weekly["rolling_mean_4"] = weekly["units_sold"].shift(1).rolling(4).mean()
weekly["week_of_year"] = weekly["week"].dt.isocalendar().week.astype(int)
weekly["is_festival"] = weekly["week_of_year"].isin([40, 41, 42, 43, 52]).astype(int)
weekly["promo_flag"] = rng.choice([0, 1], len(weekly), p=[0.8, 0.2])
weekly = weekly.dropna()

features = ["lag_1", "lag_4", "rolling_mean_4", "week_of_year", "is_festival", "promo_flag"]
X, y = weekly[features], weekly["units_sold"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
cv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
print("Time-series CV MAE:", round(-scores.mean(), 2))

# ---------------------------------------------------------------
# 6. Route optimization (pseudocode, to be implemented with OR-Tools)
# ---------------------------------------------------------------
# 1. Build distance matrix between warehouse and delivery points
# 2. Define vehicle capacity and time-window constraints
# 3. Create routing model with number of vehicles
# 4. Objective: minimize total distance (and therefore cost)
# 5. Solve using first-solution strategy + local search
# 6. Compare optimized routes vs current manual routes
