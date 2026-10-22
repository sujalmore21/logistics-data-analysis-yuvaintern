"""
Week 4: Predictive Modeling and Optimization in Logistics Systems
Author: Sujal Pradeep More

Part A  regression : predict order-to-delivery time (days)
Part B  classification : predict late-delivery risk
Part C  optimization : (1) dispatch allocation (linear programming)
                       (2) Same Day service radius
Input : cleaned_logistics_orders.csv (Week 2 output)
"""
import os, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold, cross_val_score,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              RandomForestClassifier, GradientBoostingClassifier)
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix)
from sklearn.inspection import permutation_importance
from scipy.optimize import linprog

warnings.filterwarnings("ignore")
OUT = "out/"; os.makedirs(OUT, exist_ok=True)
RS = 42
log = {}
sns.set_theme(style="whitegrid", font_scale=0.95)
PAL = {"Same Day": "#C00000", "First Class": "#ED7D31", "Second Class": "#2E5597", "Standard Class": "#548235"}
t0 = time.time()

# ------------------------------------------------------------------
# 1. DATA PREPARATION
# ------------------------------------------------------------------
df = pd.read_csv("cleaned_logistics_orders.csv", parse_dates=["order_date", "ship_date", "delivery_date"])
df["ship_delay"] = (df["ship_date"] - df["order_date"]).dt.days
df["order_dow"] = df["order_date"].dt.dayofweek
df["order_month"] = df["order_date"].dt.month
log["rows_total"] = len(df)
# delivery dates that were imputed in Week 2 are not real observations of the target
obs = df[df["delivery_date_imputed"] == 0].reset_index(drop=True)
log["rows_obs"] = len(obs); log["rows_imputed_target"] = int(len(df) - len(obs))
CAT = ["shipping_mode", "warehouse", "region", "product_category"]
NUM_A = ["distance_km", "weight_kg", "quantity", "order_month", "order_dow"]
NUM_B = NUM_A + ["ship_delay"]
TARGET = "total_lead_days"
log["target"] = {"mean": round(obs[TARGET].mean(), 2), "std": round(obs[TARGET].std(), 2),
                 "min": int(obs[TARGET].min()), "max": int(obs[TARGET].max())}

train, test = train_test_split(obs, test_size=0.2, random_state=RS)
log["n_train"] = len(train); log["n_test"] = len(test)

def prep(num):
    return ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])

class ModeMeanBaseline(BaseEstimator, RegressorMixin):
    """Baseline: predict the average delivery time of the order's shipping mode."""
    def fit(self, X, y):
        self.means_ = pd.Series(np.asarray(y), index=X["shipping_mode"].values).groupby(level=0).mean()
        self.global_ = float(np.mean(y)); return self
    def predict(self, X):
        return X["shipping_mode"].map(self.means_).fillna(self.global_).values

def metrics(y, p):
    return {"MAE": round(mean_absolute_error(y, p), 3), "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 3),
            "R2": round(r2_score(y, p), 3), "within1": round(float((np.abs(y - p) <= 1).mean() * 100), 1)}

# ------------------------------------------------------------------
# 2. PART A: REGRESSION  (two prediction points)
# ------------------------------------------------------------------
kf = KFold(5, shuffle=True, random_state=RS)
space = {
 "Decision Tree": (DecisionTreeRegressor(random_state=RS),
                   {"m__max_depth": [3, 4, 5, 6, 8, 10], "m__min_samples_leaf": [5, 20, 50]}, "grid"),
 "Random Forest": (RandomForestRegressor(random_state=RS, n_jobs=1),
                   {"m__n_estimators": [100, 150], "m__max_depth": [6, 8, 12, None],
                    "m__min_samples_leaf": [1, 3, 5, 10], "m__max_features": ["sqrt", 0.5, 1.0]}, "rand"),
 "Gradient Boosting": (GradientBoostingRegressor(random_state=RS),
                   {"m__n_estimators": [100, 200, 300], "m__learning_rate": [0.03, 0.05, 0.1],
                    "m__max_depth": [2, 3, 4], "m__subsample": [0.7, 1.0]}, "rand"),
}
res = {}; fitted = {}; preds = {}
for setname, num in [("A", NUM_A), ("B", NUM_B)]:
    feats = num + CAT
    Xtr, ytr, Xte, yte = train[feats], train[TARGET], test[feats], test[TARGET]
    res[setname] = {}
    # baseline
    b = ModeMeanBaseline().fit(Xtr, ytr); p = b.predict(Xte)
    cvb = -cross_val_score(ModeMeanBaseline(), Xtr, ytr, cv=kf, scoring="neg_mean_absolute_error")
    res[setname]["Baseline (mode average)"] = {**metrics(yte, p), "cv_mae": round(cvb.mean(), 3), "cv_std": round(cvb.std(), 3), "params": "-"}
    preds[(setname, "Baseline (mode average)")] = p
    # linear regression
    lr = Pipeline([("prep", prep(num)), ("m", LinearRegression())]).fit(Xtr, ytr); p = lr.predict(Xte)
    cvl = -cross_val_score(Pipeline([("prep", prep(num)), ("m", LinearRegression())]), Xtr, ytr, cv=kf, scoring="neg_mean_absolute_error")
    res[setname]["Linear Regression"] = {**metrics(yte, p), "cv_mae": round(cvl.mean(), 3), "cv_std": round(cvl.std(), 3), "params": "-"}
    preds[(setname, "Linear Regression")] = p; fitted[(setname, "Linear Regression")] = lr
    # tuned models
    for name, (est, grid, kind) in space.items():
        pipe = Pipeline([("prep", prep(num)), ("m", est)])
        if kind == "grid":
            s = GridSearchCV(pipe, grid, cv=kf, scoring="neg_mean_absolute_error", n_jobs=1)
        else:
            s = RandomizedSearchCV(pipe, grid, n_iter=6, cv=kf, scoring="neg_mean_absolute_error", n_jobs=1, random_state=RS)
        s.fit(Xtr, ytr); best = s.best_estimator_; p = best.predict(Xte)
        cv2 = -cross_val_score(best, Xtr, ytr, cv=KFold(5, shuffle=True, random_state=7), scoring="neg_mean_absolute_error")
        res[setname][name] = {**metrics(yte, p), "cv_mae": round(cv2.mean(), 3), "cv_std": round(cv2.std(), 3),
                              "params": {k.replace("m__", ""): (v if v is not None else "None") for k, v in s.best_params_.items()}}
        preds[(setname, name)] = p; fitted[(setname, name)] = best
        print(setname, name, res[setname][name]["MAE"], res[setname][name]["R2"], round(time.time() - t0), "s", flush=True)
log["reg"] = res
best_name = {s: min([k for k in res[s] if "Baseline" not in k], key=lambda k: res[s][k]["cv_mae"]) for s in ["A", "B"]}
log["best_model"] = best_name

# time-based hold-out: train Jan-Sep, test Oct-Dec
tb = {}
for setname, num in [("A", NUM_A), ("B", NUM_B)]:
    feats = num + CAT
    m = fitted[(setname, best_name[setname])]
    from sklearn.base import clone
    mm = clone(m)
    tr = obs[obs["order_month"] <= 9]; te = obs[obs["order_month"] > 9]
    mm.fit(tr[feats], tr[TARGET]); tb[setname] = {**metrics(te[TARGET], mm.predict(te[feats])), "n_train": len(tr), "n_test": len(te)}
log["time_split"] = tb

# permutation importance (test set)
imp = {}
for setname, num in [("A", NUM_A), ("B", NUM_B)]:
    feats = num + CAT
    m = fitted[(setname, best_name[setname])]
    pi = permutation_importance(m, test[feats], test[TARGET], n_repeats=10, random_state=RS, scoring="neg_mean_absolute_error")
    imp[setname] = pd.Series(pi.importances_mean, index=feats).sort_values()
log["importance"] = {s: imp[s].round(3).to_dict() for s in imp}

# residuals for best model B
pB = preds[("B", best_name["B"])]; pA = preds[("A", best_name["A"])]
resB = test[TARGET].values - pB; resA = test[TARGET].values - pA
log["resid"] = {"B_mean": round(float(resB.mean()), 3), "B_std": round(float(resB.std()), 3),
                "A_mean": round(float(resA.mean()), 3), "A_std": round(float(resA.std()), 3)}
mode_mae = {}
for m_ in ["Same Day", "First Class", "Second Class", "Standard Class"]:
    msk = (test["shipping_mode"] == m_).values
    mode_mae[m_] = {"A": round(float(np.abs(resA[msk]).mean()), 3), "B": round(float(np.abs(resB[msk]).mean()), 3), "n": int(msk.sum())}
log["mode_mae"] = mode_mae
log["p_Bmax_within"] = round(float((np.abs(resB) <= 1).mean() * 100), 1)

# ------------------------------------------------------------------
# 3. FIGURES FOR PART A
# ------------------------------------------------------------------
# Fig 1: framing (two prediction points)
fig, ax = plt.subplots(figsize=(9, 2.9)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
for x, t, c in [(0.2, "Order placed", "#BDD7EE"), (3.7, "Dispatched from\nwarehouse", "#BDD7EE"), (7.2, "Delivered", "#BDD7EE")]:
    ax.add_patch(plt.Rectangle((x, 1.0), 2.4, 0.9, fc=c, ec="#1F3864", lw=1.4)); ax.text(x + 1.2, 1.45, t, ha="center", va="center", fontsize=9.5)
for x1, x2 in [(2.6, 3.7), (6.1, 7.2)]:
    ax.annotate("", xy=(x2, 1.45), xytext=(x1, 1.45), arrowprops=dict(arrowstyle="->", lw=1.6, color="#1F3864"))
ax.text(3.15, 1.62, "ship_delay", ha="center", fontsize=8.5, color="#C00000"); ax.text(6.65, 1.62, "transit", ha="center", fontsize=8.5, color="#1F3864")
ax.text(1.4, 0.35, "Model A predicts here\nknown: mode, distance, weight,\nwarehouse, region, date", ha="center", fontsize=8.5, color="#2E5597")
ax.text(4.9, 0.35, "Model B predicts here\nknown: everything in A\n+ ship_delay", ha="center", fontsize=8.5, color="#548235")
ax.text(8.4, 0.35, "Target: total_lead_days\n(order to delivery)", ha="center", fontsize=8.5, color="#C00000")
ax.annotate("", xy=(1.4, 1.0), xytext=(1.4, 0.85), arrowprops=dict(arrowstyle="-", color="#2E5597"))
ax.set_title("Two prediction points for the same target", fontsize=11)
plt.tight_layout(); plt.savefig(OUT + "fig1_framing.png", dpi=170); plt.close()

# Fig 2: target and key predictor
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
sns.histplot(obs[TARGET], bins=range(0, 11), ax=axes[0], color="#2E5597", edgecolor="white")
axes[0].axvline(obs[TARGET].mean(), color="black", ls="--", label=f"mean {obs[TARGET].mean():.2f}"); axes[0].legend(fontsize=8.5)
axes[0].set_title("Target: order-to-delivery time (days)", fontsize=10.5); axes[0].set_xlabel("Days")
g = obs.groupby(["ship_delay", "shipping_mode"])[TARGET].mean().reset_index()
for m_, d_ in g.groupby("shipping_mode"):
    axes[1].plot(d_["ship_delay"], d_[TARGET], marker="o", lw=2, color=PAL[m_], label=m_)
axes[1].set_xticks([0, 1, 2]); axes[1].set_xlabel("Days from order to dispatch (ship_delay)"); axes[1].set_ylabel("Average delivery time (days)")
axes[1].set_title("Delivery time rises one day for each day of delay", fontsize=10.5); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "fig2_target.png", dpi=170); plt.close()

# Fig 3: model comparison
names = ["Baseline (mode average)", "Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
x = np.arange(len(names)); w = 0.38
for ax, key, ttl in [(axes[0], "MAE", "Test MAE (days), lower is better"), (axes[1], "R2", "Test R-squared, higher is better")]:
    va = [res["A"][n][key] for n in names]; vb = [res["B"][n][key] for n in names]
    b1 = ax.bar(x - w / 2, va, w, color="#2E5597", label="Model A (at order)"); b2 = ax.bar(x + w / 2, vb, w, color="#548235", label="Model B (at dispatch)")
    for bb, v in list(zip(b1, va)) + list(zip(b2, vb)): ax.text(bb.get_x() + bb.get_width() / 2, v + (0.01 if key == "R2" else 0.02), f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(["Baseline", "Linear", "Tree", "Forest", "Boosting"], fontsize=8.5); ax.set_title(ttl, fontsize=10.5)
axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "fig3_models.png", dpi=170); plt.close()

# Fig 4: actual vs predicted
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
rng = np.random.default_rng(1)
for ax, p, s, col in [(axes[0], pA, "A", "#2E5597"), (axes[1], pB, "B", "#548235")]:
    ax.scatter(test[TARGET] + rng.normal(0, 0.08, len(test)), p, s=7, alpha=0.3, color=col)
    ax.plot([0, 9], [0, 9], color="black", ls="--", lw=1)
    r_ = res[s][best_name[s]]
    ax.set_title(f"Model {s} ({best_name[s]})\nMAE {r_['MAE']}, R2 {r_['R2']}", fontsize=10); ax.set_xlabel("Actual days"); ax.set_ylabel("Predicted days")
    ax.set_xlim(-0.5, 9.5); ax.set_ylim(-0.5, 9.5)
plt.tight_layout(); plt.savefig(OUT + "fig4_actual_pred.png", dpi=170); plt.close()

# Fig 5: residuals
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
sns.histplot(resB, bins=30, kde=True, ax=axes[0], color="#548235", edgecolor="white")
axes[0].axvline(0, color="black", ls="--"); axes[0].set_title("Residuals of Model B (actual minus predicted)", fontsize=10); axes[0].set_xlabel("Days")
rd = pd.DataFrame({"mode": test["shipping_mode"].values, "res": resB})
sns.boxplot(data=rd, x="mode", y="res", order=list(PAL), hue="mode", palette=PAL, legend=False, ax=axes[1], fliersize=2)
axes[1].axhline(0, color="black", ls="--", lw=1); axes[1].set_title("Residuals by shipping mode (Model B)", fontsize=10); axes[1].set_xlabel(""); axes[1].set_ylabel("Days")
axes[1].tick_params(axis="x", labelsize=8)
plt.tight_layout(); plt.savefig(OUT + "fig5_residuals.png", dpi=170); plt.close()

# Fig 6: importance
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharex=False)
for ax, s, col in [(axes[0], "A", "#2E5597"), (axes[1], "B", "#548235")]:
    ser = imp[s]; ax.barh(ser.index, ser.values, color=col)
    ax.set_title(f"Model {s}: permutation importance", fontsize=10.5); ax.set_xlabel("Increase in MAE (days) when shuffled")
plt.tight_layout(); plt.savefig(OUT + "fig6_importance.png", dpi=170); plt.close()
print("regression done", round(time.time() - t0), "s", flush=True)

# ------------------------------------------------------------------
# 4. PART B: LATE-DELIVERY CLASSIFIER
# ------------------------------------------------------------------
featsB = NUM_B + CAT
ctr, cte = train_test_split(df, test_size=0.2, random_state=RS, stratify=df["is_late"])
Xc_tr, yc_tr, Xc_te, yc_te = ctr[featsB], ctr["is_late"], cte[featsB], cte["is_late"]
log["late_rate_train"] = round(float(yc_tr.mean() * 100), 1)
skf = StratifiedKFold(5, shuffle=True, random_state=RS)
clfs = {
 "Logistic Regression": Pipeline([("prep", prep(NUM_B)), ("m", LogisticRegression(max_iter=1000))]),
}
rf_s = RandomizedSearchCV(Pipeline([("prep", prep(NUM_B)), ("m", RandomForestClassifier(random_state=RS, n_jobs=1))]),
        {"m__n_estimators": [100, 150], "m__max_depth": [6, 10, None], "m__min_samples_leaf": [1, 3, 5, 10]},
        n_iter=5, cv=skf, scoring="roc_auc", n_jobs=1, random_state=RS).fit(Xc_tr, yc_tr)
gb_s = RandomizedSearchCV(Pipeline([("prep", prep(NUM_B)), ("m", GradientBoostingClassifier(random_state=RS))]),
        {"m__n_estimators": [100, 200], "m__learning_rate": [0.05, 0.1], "m__max_depth": [2, 3, 4], "m__subsample": [0.7, 1.0]},
        n_iter=5, cv=skf, scoring="roc_auc", n_jobs=1, random_state=RS).fit(Xc_tr, yc_tr)
clfs["Random Forest"] = rf_s.best_estimator_; clfs["Gradient Boosting"] = gb_s.best_estimator_
clf_res = {}; probs = {}
for n_, m_ in clfs.items():
    m_.fit(Xc_tr, yc_tr); pr = m_.predict_proba(Xc_te)[:, 1]; pd_ = (pr >= 0.5).astype(int); probs[n_] = pr
    cv_auc = cross_val_score(m_, Xc_tr, yc_tr, cv=StratifiedKFold(5, shuffle=True, random_state=7), scoring="roc_auc")
    clf_res[n_] = {"accuracy": round(accuracy_score(yc_te, pd_), 3), "precision": round(precision_score(yc_te, pd_), 3),
                   "recall": round(recall_score(yc_te, pd_), 3), "f1": round(f1_score(yc_te, pd_), 3),
                   "auc": round(roc_auc_score(yc_te, pr), 3), "cv_auc": round(cv_auc.mean(), 3), "cv_auc_std": round(cv_auc.std(), 3)}
clf_res["_params"] = {"Random Forest": {k.replace("m__", ""): (v if v is not None else "None") for k, v in rf_s.best_params_.items()},
                      "Gradient Boosting": {k.replace("m__", ""): v for k, v in gb_s.best_params_.items()}}
log["clf"] = clf_res
best_clf = max(["Logistic Regression", "Random Forest", "Gradient Boosting"], key=lambda k: clf_res[k]["auc"])
log["best_clf"] = best_clf
cm = confusion_matrix(yc_te, (probs[best_clf] >= 0.5).astype(int)); log["cm"] = cm.tolist()
log["clf_baseline_acc"] = round(float(1 - yc_te.mean()), 3)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
for n_, col in zip(clfs, ["#7F7F7F", "#2E5597", "#548235"]):
    fpr, tpr, _ = roc_curve(yc_te, probs[n_]); axes[0].plot(fpr, tpr, color=col, lw=2, label=f"{n_} (AUC {clf_res[n_]['auc']:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1); axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC curves on the test set", fontsize=10.5); axes[0].legend(fontsize=8, loc="lower right")
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1], xticklabels=["On time", "Late"], yticklabels=["On time", "Late"])
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual"); axes[1].set_title(f"Confusion matrix: {best_clf}", fontsize=10.5)
plt.tight_layout(); plt.savefig(OUT + "fig7_classifier.png", dpi=170); plt.close()

# final classifier used for optimization, trained on all data
final_clf = clfs[best_clf]; final_clf.fit(df[featsB], df["is_late"])
print("classification done", round(time.time() - t0), "s", flush=True)

# ------------------------------------------------------------------
# 5. OPTIMIZATION 1: dispatch allocation (transportation LP)
# ------------------------------------------------------------------
ns = df[df["shipping_mode"] != "Same Day"].copy()
ns["band"] = pd.cut(ns["distance_km"], [0, 250, 450, 5000], labels=["short", "medium", "long"])
ns["grp"] = ns["shipping_mode"] + " | " + ns["band"].astype(str)
groups = sorted(ns["grp"].unique()); G = len(groups); D = 3
n_g = ns.groupby("grp").size().reindex(groups).values
cap = ns["ship_delay"].value_counts().reindex([0, 1, 2]).values.astype(float)     # current capacity per delay level
P = np.zeros((G, D))
for d in range(D):
    tmp = ns.copy(); tmp["ship_delay"] = d
    tmp["p"] = final_clf.predict_proba(tmp[featsB])[:, 1]
    P[:, d] = tmp.groupby("grp")["p"].mean().reindex(groups).values
cur = ns.groupby(["grp", "ship_delay"]).size().unstack(fill_value=0).reindex(groups).reindex(columns=[0, 1, 2], fill_value=0).values.astype(float)
E = ns.groupby(["grp", "ship_delay"])["is_late"].mean().unstack().reindex(groups).reindex(columns=[0, 1, 2]).values   # empirical rates

def solve(cap_vec, mode_cap=None):
    c_ = P.flatten()
    A_eq, b_eq = [], []
    for g in range(G):
        r = np.zeros(G * D); r[g * D:(g + 1) * D] = 1; A_eq.append(r); b_eq.append(n_g[g])
    for d in range(D):
        r = np.zeros(G * D); r[d::D] = 1; A_eq.append(r); b_eq.append(cap_vec[d])
    A_ub, b_ub = None, None
    if mode_cap is not None:
        A_ub, b_ub = [], []
        for m_ in ["First Class", "Second Class", "Standard Class"]:
            r = np.zeros(G * D); tot = 0
            for g, gn in enumerate(groups):
                if gn.startswith(m_): r[g * D:(g + 1) * D] = P[g]; tot += n_g[g]
            A_ub.append(r); b_ub.append(mode_cap * tot)
    s = linprog(c_, A_ub=A_ub, b_ub=b_ub, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=(0, None), method="highs")
    return s, (s.x.reshape(G, D) if s.success else None)

late_actual_ns = int(ns["is_late"].sum())
exp_cur = float((cur * P).sum())                       # model-based, current allocation
exp_cur_emp = float(np.nansum(cur * E))                # equals the actual number of late orders
s1, X1 = solve(cap)
exp_opt = float(s1.fun); exp_opt_emp = float(np.nansum(X1 * np.nan_to_num(E)))
log["opt1"] = {"orders": int(len(ns)), "cap": cap.astype(int).tolist(), "late_actual": late_actual_ns,
               "late_model_current": round(exp_cur), "late_model_opt": round(exp_opt),
               "late_emp_opt": round(exp_opt_emp), "saved_model": round(exp_cur - exp_opt), "saved_emp": round(late_actual_ns - exp_opt_emp),
               "rate_actual": round(late_actual_ns / len(ns) * 100, 1), "rate_opt_model": round(exp_opt / len(ns) * 100, 1),
               "rate_opt_emp": round(exp_opt_emp / len(ns) * 100, 1)}
# per mode shares
def mode_share(X):
    out_ = {}
    for m_ in ["First Class", "Second Class", "Standard Class"]:
        idx = [i for i, g in enumerate(groups) if g.startswith(m_)]; tot = X[idx].sum(0); out_[m_] = (tot / tot.sum() * 100).round(1).tolist()
    return out_
log["share_cur"] = mode_share(cur); log["share_opt"] = mode_share(X1)
def mode_late(X, rates):
    o = {}
    for m_ in ["First Class", "Second Class", "Standard Class"]:
        idx = [i for i, g in enumerate(groups) if g.startswith(m_)]
        o[m_] = round(float(np.nansum(X[idx] * np.nan_to_num(rates[idx])) / X[idx].sum() * 100), 1)
    return o
log["late_mode_cur"] = mode_late(cur, E); log["late_mode_opt"] = mode_late(X1, E)
log["late_mode_opt_model"] = mode_late(X1, P)
# balanced variant: no mode may exceed 25 % predicted late
s2, X2 = solve(cap, mode_cap=0.25)
if s2.success:
    log["opt1_balanced"] = {"cap": 0.25, "late_model": round(float(s2.fun)), "rate_model": round(float(s2.fun) / len(ns) * 100, 1),
                            "late_emp": round(float(np.nansum(X2 * np.nan_to_num(E)))), "late_mode_emp": mode_late(X2, E), "share": mode_share(X2)}
else:
    log["opt1_balanced"] = None
# sensitivity to same-day dispatch capacity
N = cap.sum(); sens = []
for s0 in np.arange(0.2, 1.01, 0.1):
    cv_ = np.array([s0 * N, (1 - s0) / 2 * N, (1 - s0) / 2 * N])
    sol, Xs_ = solve(cv_)
    rnd = float(sum(n_g[g] * (cv_ / N) @ P[g] for g in range(G)))
    sens.append({"share0": round(float(s0), 2), "random": round(rnd / N * 100, 1), "optimized": round(float(sol.fun) / N * 100, 1)})
log["sens"] = sens

# Fig 8: dispatch allocation
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
modes3 = ["First Class", "Second Class", "Standard Class"]; lab = ["Dispatched same day", "1 day after", "2 days after"]
cols_ = ["#548235", "#F4B183", "#C00000"]
for ax, (ttl, sh) in zip(axes[:1], [("Share of orders by dispatch delay", None)]):
    x = np.arange(len(modes3)); wd = 0.36
    for k, (nm, shr, off, al) in enumerate([("Current", log["share_cur"], -wd / 2, 0.55), ("Optimized", log["share_opt"], wd / 2, 1.0)]):
        bottom = np.zeros(3)
        for d in range(3):
            vals = np.array([shr[m_][d] for m_ in modes3])
            ax.bar(x + off, vals, wd, bottom=bottom, color=cols_[d], alpha=al, edgecolor="white", label=lab[d] if k == 1 else None)
            for xi, v_, b_ in zip(x + off, vals, bottom):
                if v_ >= 8: ax.text(xi, b_ + v_ / 2, f"{v_:.0f}", ha="center", va="center", fontsize=7, color="white" if d != 1 else "black")
            bottom += vals
        for xi in x + off: ax.text(xi, 102, nm[:3] + ".", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(modes3, fontsize=8.5); ax.set_ylim(0, 112); ax.set_ylabel("% of orders in mode")
    ax.set_title("Who waits? Current (faded) vs optimized", fontsize=10.5); ax.legend(fontsize=7.5, loc="lower right")
o = log["opt1"]
bars = axes[1].bar(["Current\n(actual)", "Optimized\n(model estimate)", "Optimized\n(observed rates)"], [o["late_actual"], o["late_model_opt"], o["late_emp_opt"]], color=["#7F7F7F", "#2E5597", "#548235"])
for b_, v in zip(bars, [o["late_actual"], o["late_model_opt"], o["late_emp_opt"]]): axes[1].text(b_.get_x() + b_.get_width() / 2, v + 12, f"{v:,}", ha="center", fontsize=9)
axes[1].set_ylabel("Late orders (excl. Same Day)"); axes[1].set_title("Late deliveries with the same warehouse capacity", fontsize=10.5)
plt.tight_layout(); plt.savefig(OUT + "fig8_dispatch_opt.png", dpi=170); plt.close()

# Fig 9: capacity sensitivity
sdf = pd.DataFrame(sens)
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(sdf["share0"] * 100, sdf["random"], marker="o", lw=2, color="#7F7F7F", label="Orders assigned without priority")
ax.plot(sdf["share0"] * 100, sdf["optimized"], marker="o", lw=2, color="#2E5597", label="Orders prioritised by predicted risk (LP)")
cur_share0 = cap[0] / cap.sum() * 100
ax.axvline(cur_share0, color="#C00000", ls="--", lw=1.2); ax.text(cur_share0 + 1, ax.get_ylim()[1] * 0.92, "today (about 1 in 3\ndispatched same day)", fontsize=8, color="#C00000")
ax.set_xlabel("% of orders the warehouse can dispatch on the order day"); ax.set_ylabel("Expected late rate (%)")
ax.set_title("How much same-day dispatch capacity is needed?", fontsize=10.5); ax.legend(fontsize=8.5)
plt.tight_layout(); plt.savefig(OUT + "fig9_capacity.png", dpi=170); plt.close()

# ------------------------------------------------------------------
# 6. OPTIMIZATION 2: Same Day service radius
# ------------------------------------------------------------------
sd = df[df["shipping_mode"] == "Same Day"].copy()
sd["p"] = final_clf.predict_proba(sd[featsB])[:, 1]
slopes = {m_: np.polyfit(df.loc[df["shipping_mode"] == m_, "distance_km"], df.loc[df["shipping_mode"] == m_, "transport_cost"], 1) for m_ in ["Same Day", "First Class"]}
rows = []
for R in range(100, 701, 25):
    k_ = sd[sd["distance_km"] <= R]; ex = sd[sd["distance_km"] > R]
    sav = float(((np.polyval(slopes["Same Day"], ex["distance_km"])) - np.polyval(slopes["First Class"], ex["distance_km"])).sum()) if len(ex) else 0
    rows.append({"R": R, "kept_pct": len(k_) / len(sd) * 100, "late_kept": k_["is_late"].mean() * 100 if len(k_) else np.nan,
                 "late_model": k_["p"].mean() * 100 if len(k_) else np.nan, "n_ex": len(ex), "late_removed": int(ex["is_late"].sum()), "saving": sav})
rd_ = pd.DataFrame(rows); TARGET_LATE = 20.0
ok = rd_[rd_["late_kept"] <= TARGET_LATE]
Rstar = int(ok["R"].max()) if len(ok) else None
log["radius"] = {"R": Rstar, "target": TARGET_LATE, "sd_orders": len(sd), "sd_late": round(float(sd["is_late"].mean() * 100), 1),
                 **({k: (round(float(v), 1) if isinstance(v, (float, np.floating)) else int(v)) for k, v in rd_[rd_["R"] == Rstar].iloc[0].to_dict().items()} if Rstar else {})}
rd_.round(2).to_csv(OUT + "radius_table.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 3.9))
ax.plot(rd_["R"], rd_["late_kept"], color="#C00000", marker="o", ms=4, lw=2, label="Late rate of Same Day orders kept")
ax.axhline(TARGET_LATE, color="#C00000", ls=":", lw=1.2); ax.set_ylabel("Late rate (%)"); ax.set_xlabel("Maximum Same Day distance (km)")
ax2 = ax.twinx(); ax2.plot(rd_["R"], rd_["kept_pct"], color="#2E5597", marker="s", ms=4, lw=2, label="Same Day orders still offered (%)"); ax2.set_ylabel("Orders still offered (%)"); ax2.grid(False)
if Rstar: ax.axvline(Rstar, color="black", ls="--", lw=1); ax.text(Rstar + 8, ax.get_ylim()[1] * 0.1, f"{Rstar} km", fontsize=9)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
ax.set_title("Choosing a Same Day radius: service level against coverage", fontsize=10.5)
plt.tight_layout(); plt.savefig(OUT + "fig10_radius.png", dpi=170); plt.close()

# extra numbers for the narrative
log["share_rows_imputed"] = round(log["rows_imputed_target"] / log["rows_total"] * 100, 1)
log["cap_pct_today"] = round(float(cap[0] / cap.sum() * 100), 1)
log["runtime_s"] = round(time.time() - t0)
json.dump(log, open(OUT + "log.json", "w"), indent=1, default=str)
print("ALL DONE", log["runtime_s"], "s")
