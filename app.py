"""
Customer Purchase Prediction | Logistic Regression | Streamlit app
Run locally:  streamlit run app.py
Needs online_shoppers_intention.csv in the same folder (or upload it in the sidebar).
"""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve)

st.set_page_config(page_title="Customer Purchase Prediction", page_icon="🛒", layout="wide")
sns.set_theme(style="whitegrid")
RS = 42
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "online_shoppers_intention.csv")
COUNT = ["Administrative", "Informational", "ProductRelated", "ProductRelated_Duration"]
RATE = ["BounceRates", "ExitRates"]
CAT = ["Month", "VisitorType"]
BIN = ["Weekend"]
FEATURES = COUNT + RATE + CAT + BIN                    # PageValues excluded (leakage)
MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---------------------------------------------------------------- data + model
@st.cache_data(show_spinner=False)
def load(source):
    raw = pd.read_csv(source)
    n_raw, n_dup = len(raw), int(raw.duplicated().sum())
    df = raw.drop_duplicates().reset_index(drop=True)
    df["Revenue"] = df["Revenue"].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)
    return df, n_raw, n_dup

def make_model(cw="balanced", C=1.0, extra=()):
    log_scale = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("scale", StandardScaler())])
    prep = ColumnTransformer([("log_num", log_scale, COUNT + list(extra)),
                              ("num", StandardScaler(), RATE),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT),
                              ("bin", "passthrough", BIN)])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(C=C, class_weight=cw, max_iter=2000, random_state=RS))])

@st.cache_resource(show_spinner="Training the model...")
def train(df):
    X, y = df[FEATURES], df["Revenue"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    model = make_model().fit(Xtr, ytr)
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    sc = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_rows = {}
    for name, cw in {"Default weights": None, "Balanced weights": "balanced"}.items():
        r = cross_validate(make_model(cw), Xtr, ytr, cv=cv, scoring=sc)
        cv_rows[name] = {m: r[f"test_{m}"].mean() for m in sc}
    # leakage comparison
    X2 = df[FEATURES + ["PageValues"]]
    a, b, c, d = train_test_split(X2, y, test_size=0.2, stratify=y, random_state=RS)
    m2 = make_model(extra=["PageValues"]).fit(a, c)
    leak = {"p": m2.predict(b), "pr": m2.predict_proba(b)[:, 1], "y": d}
    return dict(model=model, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, prob=model.predict_proba(Xte)[:, 1],
                cv=pd.DataFrame(cv_rows).T, leak=leak)

def metrics_at(y, prob, t):
    p = (prob >= t).astype(int)
    return {"Accuracy": accuracy_score(y, p), "Precision": precision_score(y, p, zero_division=0),
            "Recall": recall_score(y, p), "F1-score": f1_score(y, p)}

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🛒 Purchase Prediction")
page = st.sidebar.radio("Go to", ["Overview", "Explore the data", "Model performance", "What drives purchases", "Try a prediction", "About"])
src = CSV if os.path.exists(CSV) else None
if src is None:
    up = st.sidebar.file_uploader("Upload online_shoppers_intention.csv", type="csv")
    if up is None:
        st.title("Customer Purchase Prediction")
        st.info("Add `online_shoppers_intention.csv` next to app.py, or upload it in the sidebar to begin.")
        st.stop()
    src = up
df, n_raw, n_dup = load(src)
T = train(df)
model, yte, prob = T["model"], T["yte"], T["prob"]

# ---------------------------------------------------------------- pages
if page == "Overview":
    st.title("Customer Purchase Prediction")
    st.caption("Logistic Regression on the UCI Online Shoppers Purchasing Intention dataset")
    pos = int(df["Revenue"].sum())
    c = st.columns(4)
    c[0].metric("Sessions (after cleaning)", f"{len(df):,}", f"-{n_dup} duplicates", delta_color="off")
    c[1].metric("Purchases", f"{pos:,}")
    c[2].metric("Purchase rate", f"{pos / len(df):.1%}")
    c[3].metric("Model inputs", len(FEATURES))
    st.subheader("Test-set performance (threshold 0.5)")
    m = metrics_at(yte, prob, 0.5); m["ROC-AUC"] = roc_auc_score(yte, prob)
    cols = st.columns(5)
    for col, (k, v) in zip(cols, m.items()): col.metric(k, f"{v:.3f}")
    st.markdown(f"""
**Goal:** predict whether a website session ends in a purchase, using browsing behaviour.  
**Why not just accuracy?** Always predicting "no purchase" is already **{1 - yte.mean():.1%}** accurate, so the model is judged on recall, F1 and ROC-AUC as well.""")
    st.subheader("Data preview")
    st.dataframe(df.head(20), width="stretch")

elif page == "Explore the data":
    st.title("Explore the data")
    t1, t2, t3, t4 = st.tabs(["Class balance", "Distributions", "Buyers vs non-buyers", "Categories & correlation"])
    with t1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.countplot(x="Revenue", data=df, ax=ax); ax.set_xticklabels(["No purchase", "Purchase"]); ax.set_xlabel("")
        st.pyplot(fig)
    with t2:
        col = st.selectbox("Feature", COUNT + RATE + ["PageValues"])
        fig, ax = plt.subplots(figsize=(7, 3.5)); sns.histplot(df[col], bins=40, ax=ax); st.pyplot(fig)
        st.caption("Counts and durations are right-skewed, which is why the model applies a log transform.")
    with t3:
        col = st.selectbox("Compare feature", COUNT + RATE, key="bx")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sns.boxplot(x="Revenue", y=col, data=df, showfliers=False, ax=ax); ax.set_xticklabels(["No purchase", "Purchase"]); st.pyplot(fig)
    with t4:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            (df.groupby("Month")["Revenue"].mean().reindex(MONTHS) * 100).plot.bar(ax=ax, ylabel="Purchase rate (%)"); st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            (df.groupby("VisitorType")["Revenue"].mean() * 100).plot.bar(ax=ax, ylabel="Purchase rate (%)"); st.pyplot(fig)
        num = COUNT + RATE + ["PageValues", "Weekend", "Revenue"]
        fig, ax = plt.subplots(figsize=(8, 6)); sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax); st.pyplot(fig)

elif page == "Model performance":
    st.title("Model performance")
    st.subheader("Cross-validation on training data (5-fold)")
    st.dataframe(T["cv"].round(4), width="stretch")
    st.caption("Default weights almost never predict a purchase (recall near zero), so balanced class weights are used.")
    st.subheader("Test set: choose a decision threshold")
    t = st.slider("Threshold", 0.10, 0.90, 0.50, 0.01)
    m = metrics_at(yte, prob, t)
    cols = st.columns(5)
    for col, (k, v) in zip(cols, m.items()): col.metric(k, f"{v:.3f}")
    cols[4].metric("ROC-AUC", f"{roc_auc_score(yte, prob):.3f}")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(confusion_matrix(yte, prob >= t), display_labels=["No", "Yes"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion matrix (t={t:.2f})"); st.pyplot(fig)
    with c2:
        f, tp, _ = roc_curve(yte, prob); fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(f, tp); ax.plot([0, 1], [0, 1], "k--"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC curve"); st.pyplot(fig)
    with c3:
        p, r, _ = precision_recall_curve(yte, prob); fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(r, p); ax.axhline(yte.mean(), color="k", ls="--"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall curve"); st.pyplot(fig)
    st.subheader("Leakage check: effect of PageValues")
    L = T["leak"]
    cmp = pd.DataFrame({"Without PageValues (final model)": {**metrics_at(yte, prob, 0.5), "ROC-AUC": roc_auc_score(yte, prob)},
                        "With PageValues": {"Accuracy": accuracy_score(L["y"], L["p"]), "Precision": precision_score(L["y"], L["p"]),
                                            "Recall": recall_score(L["y"], L["p"]), "F1-score": f1_score(L["y"], L["p"]),
                                            "ROC-AUC": roc_auc_score(L["y"], L["pr"])}})
    st.dataframe(cmp.round(4), width="stretch")
    st.caption("PageValues is closely tied to completed purchases, so it is excluded to avoid information leakage.")

elif page == "What drives purchases":
    st.title("What drives purchases")
    names = [n.split("__", 1)[1] for n in model.named_steps["prep"].get_feature_names_out()]
    coef = model.named_steps["clf"].coef_[0]
    cd = pd.DataFrame({"feature": names, "coefficient": coef, "odds_ratio": np.exp(coef)}).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(cd["feature"], cd["coefficient"], color=np.where(cd["coefficient"] > 0, "tab:green", "tab:red"))
    ax.set_xlabel("Coefficient (log-odds)"); st.pyplot(fig)
    st.dataframe(cd.round(4).reset_index(drop=True), width="stretch")
    st.caption("Numeric inputs are standardised (odds ratio = change per 1 SD). Months are compared with August, visitor types with New_Visitor. "
               "ProductRelated and ProductRelated_Duration are correlated, so interpret them together.")

elif page == "Try a prediction":
    st.title("Try a prediction")
    med = df[COUNT + RATE].median()
    c1, c2 = st.columns(2)
    with c1:
        adm = st.number_input("Administrative pages", 0, 50, int(med["Administrative"]))
        inf = st.number_input("Informational pages", 0, 50, int(med["Informational"]))
        pr = st.number_input("Product pages", 0, 800, int(med["ProductRelated"]))
        dur = st.number_input("Seconds on product pages", 0.0, 70000.0, float(round(med["ProductRelated_Duration"])))
    with c2:
        bounce = st.number_input("Bounce rate (%)", 0.0, 20.0, float(round(med["BounceRates"] * 100, 2)), step=0.1)
        exitr = st.number_input("Exit rate (%)", 0.0, 20.0, float(round(med["ExitRates"] * 100, 2)), step=0.1)
        month = st.selectbox("Month", MONTHS, index=MONTHS.index("Aug"))
        vt = st.selectbox("Visitor type", sorted(df["VisitorType"].unique()))
        wk = st.checkbox("Weekend session")
    row = pd.DataFrame([dict(Administrative=adm, Informational=inf, ProductRelated=pr, ProductRelated_Duration=dur,
                             BounceRates=bounce / 100, ExitRates=exitr / 100, Month=month, VisitorType=vt, Weekend=int(wk))])
    p = float(model.predict_proba(row)[0, 1])
    st.divider()
    st.metric("Predicted purchase probability", f"{p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))
    (st.success if p >= 0.5 else st.error)("Likely to purchase" if p >= 0.5 else "Unlikely to purchase")
    st.caption("The model uses balanced class weights, so probabilities run higher than the real purchase rate. Treat it as a ranking score. Educational demo only.")

else:
    st.title("About this project")
    st.markdown("""
**Type:** supervised learning, binary classification. **Model:** Logistic Regression only. **Target:** `Revenue` (purchase = 1).

**Inputs:** Administrative, Informational, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, Month, VisitorType, Weekend.

**Pipeline:** duplicates removed, stratified 80/20 split, log transform + standardisation for counts/durations, standardisation for rates,
one-hot encoding for categories. Everything is fitted on training data only. Class weights are balanced; 5-fold stratified cross-validation compares options.

**Limitations:** linear model; one year of one website; exit and bounce rates may not be known mid-session; `PageValues` excluded to avoid leakage.
""")
