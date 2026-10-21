"""
Week 2: Data Collection, Cleaning and Preprocessing for Logistics Analysis
Author: Sujal Pradeep More

The raw file is simulated. Its columns follow the structure of public
e-commerce / supply chain datasets (for example DataCo Smart Supply Chain),
and data quality problems were injected on purpose so that each cleaning
step has something real to fix.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
OUT = "out/"
log = {}

# =====================================================================
# 1. SIMULATED DATA COLLECTION
# =====================================================================
n = 6000
order_date = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
mode = rng.choice(["Standard Class", "Second Class", "First Class", "Same Day"], n, p=[0.55, 0.2, 0.17, 0.08])
region = rng.choice(["West", "North", "South", "East", "Central"], n, p=[0.28, 0.24, 0.22, 0.14, 0.12])
warehouse = rng.choice(["Mumbai", "Delhi", "Bengaluru", "Kolkata"], n, p=[0.3, 0.3, 0.25, 0.15])
category = rng.choice(["Electronics", "Apparel", "Home & Kitchen", "Grocery", "Books"], n, p=[0.22, 0.3, 0.2, 0.18, 0.1])
quantity = rng.integers(1, 9, n).astype(float)
weight = np.round(rng.gamma(2.2, 2.4, n) * np.sqrt(quantity), 2)
distance = np.round(rng.gamma(4.5, 70, n) + 15, 1)

ship_lag = np.where(mode == "Same Day", 0, rng.integers(0, 3, n))
transit_base = {"Standard Class": 4, "Second Class": 3, "First Class": 2, "Same Day": 0}
transit = np.array([transit_base[m] for m in mode]) + (distance / 500).round() + rng.integers(0, 2, n)
ship_date = order_date + pd.to_timedelta(ship_lag, unit="D")
delivery_date = ship_date + pd.to_timedelta(transit, unit="D")

rate = {"Standard Class": 1.6, "Second Class": 2.1, "First Class": 2.8, "Same Day": 4.2}
cost = np.round(35 + distance * np.array([rate[m] for m in mode]) * 0.55 + weight * 6 + rng.normal(0, 12, n), 2)
cost = np.clip(cost, 25, None)
promised = {"Standard Class": 7, "Second Class": 5, "First Class": 4, "Same Day": 1}
late = (delivery_date - order_date).days.values > np.array([promised[m] for m in mode])
status = np.where(late, "Late Delivery", "Delivered On Time")

raw = pd.DataFrame({
    "order_id": np.arange(100001, 100001 + n),
    "order_date": order_date, "ship_date": ship_date, "delivery_date": delivery_date,
    "shipping_mode": mode, "region": region, "warehouse": warehouse,
    "product_category": category, "quantity": quantity, "weight_kg": weight,
    "distance_km": distance, "transport_cost": cost, "delivery_status": status,
})

# ---------- inject data quality problems ----------
# (a) dates stored as text, two formats mixed
for c in ["order_date", "ship_date", "delivery_date"]:
    s = raw[c].dt.strftime("%Y-%m-%d")
    swap = rng.random(n) < 0.08
    s[swap] = raw.loc[swap, c].dt.strftime("%d/%m/%Y")
    raw[c] = s
# (b) transport cost as text with symbols for part of the rows
txt = rng.random(n) < 0.12
raw["transport_cost"] = raw["transport_cost"].astype(object)
raw.loc[txt, "transport_cost"] = raw.loc[txt, "transport_cost"].apply(lambda v: f"Rs. {v:,.2f}")
# (c) inconsistent category labels
def messy(col, p=0.07):
    idx = rng.random(n) < p
    vals = raw.loc[idx, col].astype(str)
    raw.loc[idx, col] = [rng.choice([v.lower(), v.upper(), " " + v, v + " "]) for v in vals]
for c in ["shipping_mode", "region", "warehouse", "delivery_status"]:
    messy(c)
# (d) missing values
for c, p in [("weight_kg", 0.06), ("distance_km", 0.04), ("quantity", 0.03),
             ("delivery_date", 0.025), ("product_category", 0.02), ("region", 0.015), ("transport_cost", 0.02)]:
    raw.loc[rng.random(n) < p, c] = np.nan
# (e) impossible values
bad_q = rng.choice(n, 25, replace=False); raw.loc[bad_q, "quantity"] = -raw.loc[bad_q, "quantity"].abs()
bad_d = rng.choice(n, 18, replace=False); raw.loc[bad_d, "distance_km"] = 0
# (f) impossible dates: delivery before order
bad_dt = rng.choice(n, 40, replace=False)
raw.loc[bad_dt, "delivery_date"] = raw.loc[bad_dt, "order_date"].values
raw.loc[bad_dt[:20], "delivery_date"] = (pd.to_datetime(raw.loc[bad_dt[:20], "order_date"], errors="coerce") - pd.Timedelta(days=3)).dt.strftime("%Y-%m-%d").values
# (g) outliers: entry mistakes in weight and cost
o_w = rng.choice(n, 30, replace=False); raw.loc[o_w, "weight_kg"] = rng.uniform(180, 600, 30).round(1)
o_c = rng.choice(n, 22, replace=False)
raw["transport_cost"] = raw["transport_cost"].astype(object)
raw.loc[o_c, "transport_cost"] = rng.uniform(4000, 9000, 22).round(2)
# (h) duplicate rows
raw = pd.concat([raw, raw.sample(110, random_state=3)], ignore_index=True)
raw = raw.sample(frac=1, random_state=5).reset_index(drop=True)
raw.to_csv("raw_logistics_orders.csv", index=False)

# =====================================================================
# 2. INITIAL DATA AUDIT
# =====================================================================
df = pd.read_csv("raw_logistics_orders.csv")
log["raw_shape"] = list(df.shape)
log["dtypes"] = {k: str(v) for k, v in df.dtypes.items()}
miss = df.isna().sum()
log["missing"] = {k: int(v) for k, v in miss.items() if v > 0}
log["missing_pct"] = {k: round(v / len(df) * 100, 2) for k, v in miss.items() if v > 0}
log["dup_rows"] = int(df.duplicated().sum())
log["dup_ids"] = int(df.duplicated(subset="order_id").sum())
log["label_variants_before"] = {c: int(df[c].nunique()) for c in ["shipping_mode", "region", "warehouse", "delivery_status"]}
print("RAW SHAPE", df.shape)
print(df.dtypes)
print(miss[miss > 0])

fig, ax = plt.subplots(figsize=(7, 3.6))
m = (miss[miss > 0] / len(df) * 100).sort_values()
ax.barh(m.index, m.values, color="#2E5597")
for i, v in enumerate(m.values):
    ax.text(v + 0.08, i, f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlabel("Missing values (%)"); ax.set_title("Missing values per column (raw data)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT + "fig1_missing.png", dpi=170); plt.close()

# =====================================================================
# 3. CLEANING
# =====================================================================
# 3.1 remove duplicates
before = len(df)
df = df.drop_duplicates().drop_duplicates(subset="order_id").reset_index(drop=True)
log["dups_removed"] = before - len(df)

# 3.2 fix data types: dates
def parse_dates(s):
    a = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    b = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    return a.fillna(b)
for c in ["order_date", "ship_date", "delivery_date"]:
    df[c] = parse_dates(df[c])
log["date_nat_after_parse"] = {c: int(df[c].isna().sum()) for c in ["order_date", "ship_date", "delivery_date"]}

# 3.3 fix data types: transport cost
text_cost = df["transport_cost"].astype(str).str.contains("Rs", na=False)
log["cost_text_rows"] = int(text_cost.sum())
df["transport_cost"] = pd.to_numeric(
    df["transport_cost"].astype(str).str.replace("Rs.", "", regex=False).str.replace(",", "", regex=False).str.strip(), errors="coerce")

# 3.4 standardise category labels
for c in ["shipping_mode", "region", "warehouse", "delivery_status"]:
    df[c] = df[c].str.strip().str.title()
log["label_variants_after"] = {c: int(df[c].nunique()) for c in ["shipping_mode", "region", "warehouse", "delivery_status"]}

# 3.5 impossible values -> NaN, so they are treated like missing data
neg_q = int((df["quantity"] < 0).sum()); zero_d = int((df["distance_km"] == 0).sum())
df.loc[df["quantity"] < 0, "quantity"] = np.nan
df.loc[df["distance_km"] == 0, "distance_km"] = np.nan
bad_mask = (df["delivery_date"] < df["order_date"]) | (df["delivery_date"] < df["ship_date"])
bad_dates = int(bad_mask.sum())
df.loc[bad_mask, "delivery_date"] = pd.NaT
log["neg_quantity"] = neg_q; log["zero_distance"] = zero_d; log["bad_dates"] = bad_dates

# 3.6 missing values
df["delivery_date_imputed"] = df["delivery_date"].isna().astype(int)
log["missing_before_impute"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}
df["weight_missing_flag"] = df["weight_kg"].isna().astype(int)
df["weight_kg"] = df["weight_kg"].fillna(df.groupby("product_category")["weight_kg"].transform("median"))
df["weight_kg"] = df["weight_kg"].fillna(df["weight_kg"].median())
df["distance_km"] = df["distance_km"].fillna(df.groupby("warehouse")["distance_km"].transform("median"))
df["quantity"] = df["quantity"].fillna(df["quantity"].median())
df["product_category"] = df["product_category"].fillna(df["product_category"].mode()[0])
df["region"] = df["region"].fillna(df["region"].mode()[0])
df["transport_cost"] = df["transport_cost"].fillna(df.groupby("shipping_mode")["transport_cost"].transform("median"))
# delivery date: rebuild from the typical transit time of that shipping mode
transit_days = (df["delivery_date"] - df["ship_date"]).dt.days
med_transit = transit_days.groupby(df["shipping_mode"]).transform("median")
df["delivery_date"] = df["delivery_date"].fillna(df["ship_date"] + pd.to_timedelta(med_transit, unit="D"))
log["missing_after_impute"] = int(df.drop(columns=["weight_missing_flag", "delivery_date_imputed"]).isna().sum().sum())

# 3.7 outliers (IQR rule, winsorise instead of delete)
def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75]); i = q3 - q1
    return q1 - k * i, q3 + k * i
log["outliers"] = {}
snap = df[["weight_kg", "transport_cost", "distance_km"]].copy()
for c in ["weight_kg", "transport_cost", "distance_km"]:
    lo, hi = iqr_bounds(df[c])
    cnt = int(((df[c] < lo) | (df[c] > hi)).sum())
    log["outliers"][c] = {"lower": round(lo, 2), "upper": round(hi, 2), "count": cnt,
                          "max_before": round(float(df[c].max()), 2)}
# z-score cross check on weight
z = (df["weight_kg"] - df["weight_kg"].mean()) / df["weight_kg"].std()
log["z_gt3_weight"] = int((z.abs() > 3).sum())
# extreme entry mistakes (above 99.9th percentile style) are capped at the IQR upper fence
for c in ["weight_kg", "transport_cost"]:
    lo, hi = iqr_bounds(df[c], k=3.0)       # 3*IQR fence = extreme outliers only
    log["outliers"][c]["extreme_count"] = int(((df[c] < lo) | (df[c] > hi)).sum())
    log["outliers"][c]["extreme_upper"] = round(hi, 2)
    df[c] = df[c].clip(lo, hi)
    log["outliers"][c]["max_after"] = round(float(df[c].max()), 2)

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
for ax, c, t in zip(axes, ["weight_kg", "transport_cost"], ["Weight (kg)", "Transport cost"]):
    ax.boxplot([snap[c], df[c]], labels=["Before", "After"], patch_artist=True,
               boxprops=dict(facecolor="#BDD7EE"), medianprops=dict(color="#C00000"))
    ax.set_title(t); ax.spines[["top", "right"]].set_visible(False)
plt.suptitle("Outlier treatment: boxplots before and after capping", fontsize=11)
plt.tight_layout(); plt.savefig(OUT + "fig2_outliers.png", dpi=170); plt.close()

# 3.8 consistency / derived features
df["transit_days"] = (df["delivery_date"] - df["ship_date"]).dt.days
df["total_lead_days"] = (df["delivery_date"] - df["order_date"]).dt.days
df["cost_per_km"] = (df["transport_cost"] / df["distance_km"]).round(3)
df["order_month"] = df["order_date"].dt.month
df["is_late"] = (df["delivery_status"] == "Late Delivery").astype(int)
log["neg_transit_after"] = int((df["transit_days"] < 0).sum())

# 3.9 normalisation
num = ["weight_kg", "distance_km", "transport_cost", "quantity"]
for c in num:
    mn, mx = df[c].min(), df[c].max()
    df[c + "_minmax"] = (df[c] - mn) / (mx - mn)
    df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()
log["minmax_range"] = {c: [round(float(df[c + "_minmax"].min()), 2), round(float(df[c + "_minmax"].max()), 2)] for c in num}
log["z_stats"] = {c: [round(float(df[c + "_z"].mean()), 3), round(float(df[c + "_z"].std()), 3)] for c in num}
log["scale_before"] = {c: [round(float(df[c].min()), 2), round(float(df[c].max()), 2)] for c in num}

fig, axes = plt.subplots(1, 3, figsize=(9, 3.1))
axes[0].hist(df["distance_km"], bins=35, color="#2E5597"); axes[0].set_title("Original (km)")
axes[1].hist(df["distance_km_minmax"], bins=35, color="#548235"); axes[1].set_title("Min-Max (0 to 1)")
axes[2].hist(df["distance_km_z"], bins=35, color="#BF8F00"); axes[2].set_title("Z-score")
for a in axes: a.spines[["top", "right"]].set_visible(False)
plt.suptitle("Distance before and after scaling (shape stays the same)", fontsize=11)
plt.tight_layout(); plt.savefig(OUT + "fig3_scaling.png", dpi=170); plt.close()

# 3.10 encoding
enc = pd.get_dummies(df[["shipping_mode", "region", "warehouse", "product_category"]], drop_first=True, dtype=int)
log["encoded_cols"] = int(enc.shape[1])
final = pd.concat([df, enc], axis=1)
log["final_shape"] = list(final.shape)

# =====================================================================
# 4. VALIDATION
# =====================================================================
checks = {
    "No duplicate order_id": bool(final["order_id"].is_unique),
    "No missing values in core columns": bool(final[["order_date","ship_date","delivery_date","weight_kg","distance_km","transport_cost","quantity","region","product_category"]].isna().sum().sum() == 0),
    "All quantities positive": bool((final["quantity"] > 0).all()),
    "All distances positive": bool((final["distance_km"] > 0).all()),
    "Delivery date never before order date": bool((final["total_lead_days"] >= 0).all()),
    "Date columns are datetime64": all(str(final[c].dtype).startswith("datetime64") for c in ["order_date","ship_date","delivery_date"]),
}
log["checks"] = checks
for k, v in checks.items(): print(("PASS" if v else "FAIL"), k)

# impact on summary stats (raw numeric vs cleaned)
raw_num = pd.read_csv("raw_logistics_orders.csv")
raw_num["transport_cost"] = pd.to_numeric(raw_num["transport_cost"].astype(str).str.replace("Rs.", "", regex=False).str.replace(",", "", regex=False).str.strip(), errors="coerce")
log["impact"] = {c: {"raw_mean": round(float(raw_num[c].mean()), 2), "clean_mean": round(float(final[c].mean()), 2),
                     "raw_std": round(float(raw_num[c].std()), 2), "clean_std": round(float(final[c].std()), 2)}
                 for c in ["weight_kg", "transport_cost"]}
log["late_rate"] = round(float(final["is_late"].mean() * 100), 1)
log["lead_by_mode"] = {k: round(float(v), 2) for k, v in final.groupby("shipping_mode")["total_lead_days"].mean().items()}
final.to_csv("cleaned_logistics_orders.csv", index=False)
json.dump(log, open(OUT + "log.json", "w"), indent=1)
print(json.dumps(log, indent=1)[:6000])
