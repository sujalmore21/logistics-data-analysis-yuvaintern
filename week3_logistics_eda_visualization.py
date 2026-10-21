"""
Week 3: Advanced Data Analysis and Visualization in Logistics
Author: Sujal Pradeep More

Input : cleaned_logistics_orders.csv (output of the Week 2 cleaning pipeline)
Output: figures (PNG) in out/ and a log.json with every number used in the report
"""
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import os
OUT = "out/"
os.makedirs(OUT, exist_ok=True)
log = {}
sns.set_theme(style="whitegrid", font_scale=0.95)
MODE_ORDER = ["Same Day", "First Class", "Second Class", "Standard Class"]
PAL = {"Same Day": "#C00000", "First Class": "#ED7D31", "Second Class": "#2E5597", "Standard Class": "#548235"}

df = pd.read_csv("cleaned_logistics_orders.csv", parse_dates=["order_date", "ship_date", "delivery_date"])
df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
log["rows"] = len(df)
log["period"] = [str(df["order_date"].min().date()), str(df["order_date"].max().date())]

# ------------------------------------------------------------------
# 1. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------------
cols = ["quantity", "weight_kg", "distance_km", "transport_cost", "cost_per_km", "total_lead_days"]
desc = {}
for c in cols:
    s = df[c]
    desc[c] = {"mean": round(s.mean(), 2), "median": round(s.median(), 2),
               "mode": round(float(s.mode().iloc[0]), 2), "std": round(s.std(), 2),
               "min": round(s.min(), 2), "max": round(s.max(), 2), "skew": round(float(stats.skew(s)), 2)}
log["desc"] = desc
log["late_rate"] = round(df["is_late"].mean() * 100, 1)
log["total_cost"] = round(df["transport_cost"].sum(), 0)
log["avg_cost"] = round(df["transport_cost"].mean(), 2)
print(pd.DataFrame(desc).T)

# FIG 1 distributions
fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
for ax, (c, t, col) in zip(axes.ravel(), [("transport_cost", "Transport cost per order (Rs.)", "#2E5597"),
                                          ("total_lead_days", "Order-to-delivery time (days)", "#548235"),
                                          ("distance_km", "Delivery distance (km)", "#ED7D31"),
                                          ("weight_kg", "Shipment weight (kg)", "#7F6000")]):
    if c == "total_lead_days":
        sns.histplot(df[c], bins=range(0, int(df[c].max()) + 2), ax=ax, color=col, edgecolor="white")
    else:
        sns.histplot(df[c], bins=40, kde=True, ax=ax, color=col, edgecolor="white")
    ax.axvline(df[c].mean(), color="black", ls="--", lw=1.2, label=f"mean {df[c].mean():.1f}")
    ax.axvline(df[c].median(), color="#C00000", ls="-", lw=1.2, label=f"median {df[c].median():.1f}")
    ax.set_title(t, fontsize=10.5); ax.set_xlabel(""); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "fig1_distributions.png", dpi=170); plt.close()

# ------------------------------------------------------------------
# 2. SHIPPING MODE: cost and lead time
# ------------------------------------------------------------------
g = df.groupby("shipping_mode").agg(orders=("order_id", "count"), avg_cost=("transport_cost", "mean"),
        avg_lead=("total_lead_days", "mean"), avg_cpk=("cost_per_km", "mean"),
        late=("is_late", "mean"), avg_dist=("distance_km", "mean")).reindex(MODE_ORDER)
g["late"] = g["late"] * 100
g["share"] = g["orders"] / g["orders"].sum() * 100
log["mode"] = g.round(2).to_dict("index")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
sns.boxplot(data=df, x="shipping_mode", y="transport_cost", order=MODE_ORDER, hue="shipping_mode", palette=PAL, legend=False, ax=axes[0], fliersize=2)
axes[0].set_title("Transport cost by shipping mode", fontsize=10.5); axes[0].set_xlabel(""); axes[0].set_ylabel("Cost (Rs.)")
sns.boxplot(data=df, x="shipping_mode", y="total_lead_days", order=MODE_ORDER, hue="shipping_mode", palette=PAL, legend=False, ax=axes[1], fliersize=2)
axes[1].set_title("Delivery time by shipping mode", fontsize=10.5); axes[1].set_xlabel(""); axes[1].set_ylabel("Days")
for a in axes: a.tick_params(axis="x", labelsize=8)
plt.tight_layout(); plt.savefig(OUT + "fig2_mode_box.png", dpi=170); plt.close()

# ------------------------------------------------------------------
# 3. CORRELATION
# ------------------------------------------------------------------
cc = ["distance_km", "weight_kg", "quantity", "transport_cost", "cost_per_km", "total_lead_days", "transit_days"]
corr = df[cc].corr()
log["corr"] = corr.round(2).to_dict()
fig, ax = plt.subplots(figsize=(7, 5.4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8}, annot_kws={"size": 8.5})
ax.set_title("Correlation matrix of numeric logistics variables", fontsize=11)
plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.savefig(OUT + "fig3_corr.png", dpi=170); plt.close()

# ------------------------------------------------------------------
# 4. COST DRIVERS: scatter + regression
# ------------------------------------------------------------------
samp = df.sample(1800, random_state=1)
fig, ax = plt.subplots(figsize=(8, 4.8))
for m in MODE_ORDER:
    s = samp[samp["shipping_mode"] == m]
    ax.scatter(s["distance_km"], s["transport_cost"], s=9, alpha=0.35, color=PAL[m], label=m)
    k, b = np.polyfit(df.loc[df["shipping_mode"] == m, "distance_km"], df.loc[df["shipping_mode"] == m, "transport_cost"], 1)
    xs = np.linspace(df["distance_km"].min(), df["distance_km"].max(), 50)
    ax.plot(xs, k * xs + b, color=PAL[m], lw=2)
    log.setdefault("slope", {})[m] = round(k, 3)
ax.set_xlabel("Distance (km)"); ax.set_ylabel("Transport cost (Rs.)")
ax.set_title("Cost rises with distance, and the slope depends on the shipping mode", fontsize=10.5)
ax.legend(markerscale=2.5, fontsize=8.5); plt.tight_layout(); plt.savefig(OUT + "fig4_scatter.png", dpi=170); plt.close()

# multiple regression with standardised coefficients
X = pd.get_dummies(df[["distance_km", "weight_kg", "quantity", "shipping_mode"]], columns=["shipping_mode"], drop_first=True, dtype=float)
y = df["transport_cost"]
Xs = (X - X.mean()) / X.std(); ys = (y - y.mean()) / y.std()
A = np.column_stack([np.ones(len(Xs)), Xs.values])
beta, *_ = np.linalg.lstsq(A, ys.values, rcond=None)
pred = A @ beta
r2 = 1 - ((ys.values - pred) ** 2).sum() / ((ys.values - ys.mean()) ** 2).sum()
coef = pd.Series(beta[1:], index=X.columns).sort_values()
log["r2"] = round(float(r2), 3); log["coef"] = coef.round(3).to_dict()
raw_beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X.values]), y.values, rcond=None)
log["raw_coef"] = dict(zip(["intercept"] + list(X.columns), np.round(raw_beta, 2).tolist()))
fig, ax = plt.subplots(figsize=(7.5, 3.6))
cols_ = ["#2E5597" if v > 0 else "#C00000" for v in coef.values]
ax.barh([c.replace("shipping_mode_", "mode: ") for c in coef.index], coef.values, color=cols_)
for i, v in enumerate(coef.values): ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
ax.set_xlabel("Standardised coefficient (effect on cost in standard deviations)")
ax.set_title(f"What drives transport cost? (linear model, R-squared = {r2:.2f})", fontsize=10.5)
plt.tight_layout(); plt.savefig(OUT + "fig5_drivers.png", dpi=170); plt.close()

# ------------------------------------------------------------------
# 5. TIME TREND
# ------------------------------------------------------------------
mt = df.groupby("month").agg(orders=("order_id", "count"), avg_cost=("transport_cost", "mean"), late=("is_late", "mean"))
mt["late"] *= 100
log["monthly"] = {str(k.date())[:7]: {"orders": int(v.orders), "avg_cost": round(v.avg_cost, 1), "late": round(v.late, 1)} for k, v in mt.iterrows()}
log["month_peak"] = str(mt["orders"].idxmax().date())[:7]; log["month_low"] = str(mt["orders"].idxmin().date())[:7]
log["month_orders_range"] = [int(mt["orders"].min()), int(mt["orders"].max())]
log["month_cv"] = round(float(mt["orders"].std() / mt["orders"].mean() * 100), 1)
fig, ax = plt.subplots(figsize=(8.5, 3.8))
ax.bar(mt.index, mt["orders"], width=20, color="#BDD7EE", label="Orders")
ax.set_ylabel("Orders per month"); ax.set_ylim(0, mt["orders"].max() * 1.25)
ax2 = ax.twinx(); ax2.plot(mt.index, mt["late"], color="#C00000", marker="o", lw=2, label="Late deliveries (%)")
ax2.set_ylabel("Late deliveries (%)"); ax2.set_ylim(0, max(40, mt["late"].max() * 1.3)); ax2.grid(False)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b")); ax.set_xticks(mt.index)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, fontsize=8.5)
ax.set_title("Monthly order volume and late-delivery rate, 2025", fontsize=10.5)
plt.tight_layout(); plt.savefig(OUT + "fig6_monthly.png", dpi=170); plt.close()

# ------------------------------------------------------------------
# 6. LATE DELIVERIES: where do they come from?
# ------------------------------------------------------------------
lw = df.groupby("warehouse")["is_late"].agg(["mean", "count"]); lw["mean"] *= 100
lr = df.groupby("region")["is_late"].agg(["mean", "count"]); lr["mean"] *= 100
log["late_wh"] = lw["mean"].round(1).to_dict(); log["late_reg"] = lr["mean"].round(1).to_dict()
chi_wh = stats.chi2_contingency(pd.crosstab(df["warehouse"], df["is_late"]))
chi_rg = stats.chi2_contingency(pd.crosstab(df["region"], df["is_late"]))
chi_md = stats.chi2_contingency(pd.crosstab(df["shipping_mode"], df["is_late"]))
log["chi"] = {"warehouse": round(float(chi_wh[1]), 3), "region": round(float(chi_rg[1]), 3), "mode": float(f"{chi_md[1]:.2e}")}
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.5), sharey=True)
for ax, (ser, t, order, colr) in zip(axes, [(g["late"], "Shipping mode", MODE_ORDER, [PAL[m] for m in MODE_ORDER]),
                                           (lw["mean"].sort_values(ascending=False), "Warehouse", None, "#2E5597"),
                                           (lr["mean"].sort_values(ascending=False), "Region", None, "#2E5597")]):
    ser = ser.loc[order] if order else ser
    bars = ax.bar(ser.index, ser.values, color=colr)
    ax.axhline(df["is_late"].mean() * 100, color="black", ls="--", lw=1)
    for b_, v in zip(bars, ser.values): ax.text(b_.get_x() + b_.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_title(t, fontsize=10.5); ax.tick_params(axis="x", labelsize=7.5, rotation=20)
axes[0].set_ylabel("Late deliveries (%)")
fig.suptitle("Late-delivery rate by group (dashed line = overall average)", fontsize=11)
plt.tight_layout(); plt.savefig(OUT + "fig7_late_groups.png", dpi=170); plt.close()

pt = df.pivot_table(index="warehouse", columns="shipping_mode", values="is_late", aggfunc="mean")[MODE_ORDER] * 100
log["late_wh_mode"] = pt.round(1).to_dict("index")
fig, ax = plt.subplots(figsize=(7, 3.4))
sns.heatmap(pt, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, linewidths=0.5, cbar_kws={"label": "Late (%)"})
ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("Late-delivery rate (%): warehouse x shipping mode", fontsize=10.5)
plt.tight_layout(); plt.savefig(OUT + "fig8_heat.png", dpi=170); plt.close()

# ------------------------------------------------------------------
# 7. DISTANCE BANDS
# ------------------------------------------------------------------
bins = [0, 200, 400, 600, 800, 2000]; labs = ["0-200", "200-400", "400-600", "600-800", "800+"]
df["band"] = pd.cut(df["distance_km"], bins=bins, labels=labs)
bd = df.groupby("band", observed=True).agg(orders=("order_id", "count"), lead=("total_lead_days", "mean"),
        late=("is_late", "mean"), cpk=("cost_per_km", "mean"), cost=("transport_cost", "mean"))
bd["late"] *= 100
log["band"] = bd.round(2).to_dict("index")
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(bd.index.astype(str), bd["late"], color="#F4B183", label="Late (%)")
for i, v in enumerate(bd["late"]): ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=8.5)
ax.set_ylabel("Late deliveries (%)"); ax.set_xlabel("Distance band (km)"); ax.set_ylim(0, bd["late"].max() * 1.3)
ax2 = ax.twinx(); ax2.plot(bd.index.astype(str), bd["cpk"], color="#1F3864", marker="s", lw=2, label="Cost per km (Rs.)")
ax2.set_ylabel("Average cost per km (Rs.)"); ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8.5, ncol=2)
ax.set_title("Longer routes: more late deliveries, but cheaper per km", fontsize=10.5)
plt.tight_layout(); plt.savefig(OUT + "fig9_bands.png", dpi=170); plt.close()

# ------------------------------------------------------------------
# 8. EXTRA TABLES
# ------------------------------------------------------------------
cat = df.groupby("product_category").agg(orders=("order_id", "count"), cost=("transport_cost", "mean"), late=("is_late", "mean"))
cat["late"] *= 100
log["cat"] = cat.round(2).to_dict("index")
log["cost_by_cat_spread"] = round(float(cat["cost"].max() - cat["cost"].min()), 2)
top = df.groupby("shipping_mode")["transport_cost"].sum().reindex(MODE_ORDER)
log["cost_share_mode"] = (top / top.sum() * 100).round(1).to_dict()
log["corr_dist_cost"] = round(float(df["distance_km"].corr(df["transport_cost"])), 2)
# Same Day detail
sd = df[df["shipping_mode"] == "Same Day"]
log["sameday_late_short"] = round(float(sd[sd["distance_km"] <= 300]["is_late"].mean() * 100), 1)
log["sameday_late_long"] = round(float(sd[sd["distance_km"] > 300]["is_late"].mean() * 100), 1)
std_ = df[df["shipping_mode"] == "Standard Class"]
log["std_late_short"] = round(float(std_[std_["distance_km"] <= 300]["is_late"].mean() * 100), 1)
log["std_late_long"] = round(float(std_[std_["distance_km"] > 300]["is_late"].mean() * 100), 1)
# ship delay (order -> ship) share of lead time
df["ship_delay"] = (df["ship_date"] - df["order_date"]).dt.days
log["ship_delay_mean"] = round(float(df["ship_delay"].mean()), 2)
log["late_by_shipdelay"] = (df.groupby("ship_delay")["is_late"].mean() * 100).round(1).to_dict()
log["late_orders"] = int(df["is_late"].sum())

# ------------------------------------------------------------------
# 9. DISPATCH DELAY BOTTLENECK (order -> ship), excluding Same Day
# ------------------------------------------------------------------
ns = df[df["shipping_mode"] != "Same Day"]
dd = ns.pivot_table(index="ship_delay", columns="shipping_mode", values="is_late", aggfunc="mean")[["First Class", "Second Class", "Standard Class"]] * 100
cnt = ns.groupby("ship_delay").size()
log["dispatch"] = dd.round(1).to_dict("index"); log["dispatch_n"] = cnt.to_dict()
log["dispatch_all"] = (ns.groupby("ship_delay")["is_late"].mean() * 100).round(1).to_dict()
log["dispatch_share"] = (cnt / cnt.sum() * 100).round(1).to_dict()
log["late_if_delay2_share_of_late"] = round(float(ns[(ns.ship_delay == 2)]["is_late"].sum() / ns["is_late"].sum() * 100), 1)
fig, ax = plt.subplots(figsize=(8, 3.8))
w = 0.26
for i, m in enumerate(["First Class", "Second Class", "Standard Class"]):
    ax.bar(np.arange(3) + (i - 1) * w, dd[m].values, width=w, color=PAL[m], label=m)
    for x_, v in zip(np.arange(3) + (i - 1) * w, dd[m].values): ax.text(x_, v + 1, f"{v:.0f}", ha="center", fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels(["Shipped same day\nas order", "1 day after order", "2 days after order"])
ax.set_ylabel("Late deliveries (%)"); ax.legend(fontsize=8.5)
ax.set_title("Dispatch delay is the clearest bottleneck (Same Day mode excluded)", fontsize=10.5)
plt.tight_layout(); plt.savefig(OUT + "fig10_dispatch.png", dpi=170); plt.close()
json.dump(log, open(OUT + "log.json", "w"), indent=1, default=str)

# what-if: delay-2 orders get the late rate of delay-1 orders of the same mode (rough estimate)
rate1 = ns[ns.ship_delay == 1].groupby("shipping_mode")["is_late"].mean()
n2 = ns[ns.ship_delay == 2].groupby("shipping_mode")["is_late"].agg(["sum", "count"])
saved = float((n2["sum"] - n2["count"] * rate1.reindex(n2.index)).sum())
log["whatif_saved"] = int(round(saved)); log["whatif_new_rate"] = round(float((df["is_late"].sum() - saved) / len(df) * 100), 1)
log["delay2_orders"] = int(n2["count"].sum())
json.dump(log, open(OUT + "log.json", "w"), indent=1, default=str)

print("done")
