import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# ---------------------------------------------------------
# PAGE + THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="QUANTUM | Customer Churn & Risk Scoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root { color-scheme: dark; }
    .stApp { background: #111518; color: #edf1f3; }
    [data-testid="stSidebar"] { background: #171d21; border-right: 1px solid #30383e; }
    [data-testid="stHeader"] { background: rgba(17,21,24,0.95); }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg,#20272c,#191f23);
        border: 1px solid #343d43; padding: 14px 16px; border-radius: 12px;
    }
    [data-testid="stMetricLabel"] { color: #b8c1c7; }
    [data-testid="stMetricValue"] { color: #f4f6f7; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #1a2024; border: 1px solid #333c42; border-radius: 12px;
    }
    .main-title { font-size: 27px; font-weight: 750; letter-spacing: .4px; }
    .subtle { color: #9daab2; font-size: 13px; }
    .section-title { font-size: 17px; font-weight: 700; margin: 2px 0 12px 0; }
    .risk-high { color: #ff7043; font-weight: 700; }
    .risk-medium { color: #ffb74d; font-weight: 700; }
    .risk-low { color: #82a9c8; font-weight: 700; }
    .stButton>button, .stDownloadButton>button {
        background: #e96b32; color: white; border: 0; border-radius: 8px;
        font-weight: 650;
    }
    .stButton>button:hover, .stDownloadButton>button:hover { background: #ff8248; color: white; }
    div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
        background: #20272c; color: #edf1f3;
    }
    hr { border-color: #30383e; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA + PREPROCESSING
# ---------------------------------------------------------
TARGET = "Exited"
DROP_COLUMNS = ["Surname", "CustomerId", "RowNumber"]

@st.cache_data(show_spinner=False)
def load_data():
    # Load the Excel file from the same folder as this Python script.
    file_path = Path(__file__).resolve().parent / "European_Bank.xlsx"

    if not file_path.exists():
        st.error(
            "Dataset not found. Place 'European_Bank.xlsx' in the same folder "
            f"as this script. Expected location: {file_path}"
        )
        st.stop()

    return pd.read_excel(file_path)

@st.cache_data(show_spinner=False)
def prepare_features(raw_df):
    data = raw_df.copy()
    required = {TARGET, "Age", "Tenure", "Balance", "Estimated_Salary",
                "Num_Of_Products", "Is_Active_Member"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(sorted(missing)))

    # Engineered features are calculated before the train/test split.
    salary = data["Estimated_Salary"].replace(0, np.nan)
    data["Balance_Salary_Ratio"] = (data["Balance"] / salary).replace([np.inf, -np.inf], np.nan)
    data["Product_Density"] = data["Num_Of_Products"] / (data["Tenure"] + 1)
    data["Engagement_Product"] = data["Is_Active_Member"] * data["Num_Of_Products"]
    data["Age_Tenure"] = data["Age"] * data["Tenure"]

    y = pd.to_numeric(data[TARGET], errors="coerce")
    usable = y.notna()
    data = data.loc[usable].copy()
    y = y.loc[usable].astype(int)

    drop_now = [c for c in DROP_COLUMNS + [TARGET] if c in data.columns]
    X = data.drop(columns=drop_now)
    # Remove columns with no usable values; imputers handle remaining missing data.
    X = X.dropna(axis=1, how="all")
    return X, y, data

@st.cache_resource(show_spinner="Training churn models...")
def train_models(X_train, y_train):
    categorical = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X_train.columns if c not in categorical]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])

    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=350, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        )
    }
    fitted = {}
    for name, estimator in estimators.items():
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
    return fitted

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def risk_band(prob):
    if prob >= 0.60:
        return "High"
    if prob >= 0.30:
        return "Medium"
    return "Low"

def get_feature_names(pipeline):
    try:
        return pipeline.named_steps["prep"].get_feature_names_out()
    except Exception:
        return np.array([f"Feature {i+1}" for i in range(
            pipeline.named_steps["model"].n_features_in_
        )])

def get_importance(pipeline):
    model = pipeline.named_steps["model"]
    names = get_feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])
    result = pd.DataFrame({"Feature": names, "Importance": values})
    result["Feature"] = result["Feature"].str.replace(r"^(num|cat)__", "", regex=True)
    return result.sort_values("Importance", ascending=False)

def styled_chart(fig):
    fig.patch.set_facecolor("#1a2024")
    for ax in fig.axes:
        ax.set_facecolor("#1a2024")
        ax.tick_params(colors="#c4cdd2")
        ax.xaxis.label.set_color("#c4cdd2")
        ax.yaxis.label.set_color("#c4cdd2")
        ax.title.set_color("#f1f4f5")
        for spine in ax.spines.values():
            spine.set_color("#3a444b")
        ax.grid(color="#343d43", alpha=0.55)
    return fig

def customer_label(row, index):
    if "CustomerId" in row.index:
        return str(row["CustomerId"])
    return f"Customer {index + 1}"

# ---------------------------------------------------------
# SIDEBAR + LOAD
# ---------------------------------------------------------
st.sidebar.markdown("## ◉ QUANTUM")
st.sidebar.caption("BANK CUSTOMER CHURN & RISK SCORING")
model_choice = st.sidebar.selectbox(
    "Prediction model",
    ["Random Forest", "Gradient Boosting", "Logistic Regression"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Risk bands: High ≥ 60% · Medium 30–59% · Low < 30%")
st.sidebar.caption("Models are trained on an 80/20 stratified split.")

try:
    raw = load_data()
    X, y, enriched = prepare_features(raw)
except Exception as exc:
    st.error(f"Could not load or prepare the dataset: {exc}")
    st.info("Place European_Bank.xlsx beside this Python file, or upload it using the sidebar.")
    st.stop()

if y.nunique() < 2:
    st.error("The target column 'Exited' must contain both 0 and 1 classes.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
models = train_models(X_train, y_train)
model = models[model_choice]

# Predictions and evaluation are calculated once for the selected model.
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= 0.50).astype(int)
auc = roc_auc_score(y_test, test_prob)
accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, zero_division=0)
recall = recall_score(y_test, test_pred, zero_division=0)
f1 = f1_score(y_test, test_pred, zero_division=0)

all_prob = model.predict_proba(X)[:, 1]
risk = pd.DataFrame({
    "Risk Score": np.round(all_prob * 100).astype(int),
    "Churn Probability": all_prob,
    "Risk Level": [risk_band(p) for p in all_prob]
}, index=X.index)

# ---------------------------------------------------------
# HEADER + KPI ROW
# ---------------------------------------------------------
header_left, header_right = st.columns([4, 1])
with header_left:
    st.markdown('<div class="main-title">BANK CUSTOMER CHURN & RISK SCORING</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Predictive intelligence · Customer retention · Model monitoring</div>', unsafe_allow_html=True)
with header_right:
    st.markdown(f"<div style='text-align:right;color:#aeb9bf;padding-top:8px'>MODEL: <b>{model_choice}</b></div>", unsafe_allow_html=True)

st.markdown("---")
total_customers = len(X)
churn_rate = float(y.mean() * 100)
high_count = int((risk["Risk Level"] == "High").sum())
avg_risk = float(risk["Risk Score"].mean())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Customers", f"{total_customers:,}")
k2.metric("Historical Churn Rate", f"{churn_rate:.1f}%")
k3.metric("High-Risk Customers", f"{high_count:,}")
k4.metric("Average Risk Score", f"{avg_risk:.1f}/100")
k5.metric("ROC-AUC", f"{auc:.3f}")

# ---------------------------------------------------------
# MAIN VISUAL ANALYTICS
# ---------------------------------------------------------
left, middle, right = st.columns([1.05, 1.15, 1.05], gap="medium")

with left:
    with st.container(border=True):
        st.markdown('<div class="section-title">CHURN RISK BREAKDOWN</div>', unsafe_allow_html=True)
        counts = risk["Risk Level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 3.3))
        bars = ax.bar(counts.index, counts.values, color=["#ef6535", "#f5a342", "#526b82"], width=0.58)
        ax.set_ylabel("Customers")
        ax.set_title("Customers by predicted risk")
        for bar, value in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:,}",
                    ha="center", va="bottom", color="#f1f4f5", fontsize=9)
        st.pyplot(styled_chart(fig), use_container_width=True)
        plt.close(fig)

with middle:
    with st.container(border=True):
        st.markdown('<div class="section-title">GLOBAL FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
        importance = get_importance(model).head(8).sort_values("Importance")
        if not importance.empty:
            fig, ax = plt.subplots(figsize=(5, 3.3))
            ax.barh(importance["Feature"], importance["Importance"], color="#ef793d")
            ax.set_xlabel("Relative importance")
            ax.set_title("Top model drivers")
            st.pyplot(styled_chart(fig), use_container_width=True)
            plt.close(fig)
        else:
            st.info("Feature importance is not available for this model.")

with right:
    with st.container(border=True):
        st.markdown('<div class="section-title">CUSTOMER RISK SEGMENTS</div>', unsafe_allow_html=True)
        segment_counts = risk["Risk Level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
        fig, ax = plt.subplots(figsize=(4.5, 3.3))
        ax.pie(
            segment_counts.values, labels=segment_counts.index,
            colors=["#ef6535", "#f5a342", "#526b82"],
            startangle=90, counterclock=False,
            wedgeprops={"width": 0.42, "edgecolor": "#1a2024"},
            autopct=lambda p: f"{p:.0f}%" if p >= 4 else ""
        )
        ax.set_title("Predicted portfolio mix")
        st.pyplot(styled_chart(fig), use_container_width=True)
        plt.close(fig)

# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------
perf_left, perf_right = st.columns([1.25, 1], gap="medium")
with perf_left:
    with st.container(border=True):
        st.markdown('<div class="section-title">ROC CURVE</div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, test_prob)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(fpr, tpr, color="#f0783b", linewidth=2.5, label=f"{model_choice} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#7c858b", label="Random baseline")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(facecolor="#20272c", edgecolor="#3a444b", labelcolor="#edf1f3", fontsize=8)
        st.pyplot(styled_chart(fig), use_container_width=True)
        plt.close(fig)

with perf_right:
    with st.container(border=True):
        st.markdown('<div class="section-title">MODEL EVALUATION</div>', unsafe_allow_html=True)
        metric_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            "Score": [accuracy, precision, recall, f1, auc]
        })
        st.dataframe(
            metric_df.style.format({"Score": "{:.3f}"}).background_gradient(
                subset=["Score"], cmap="Oranges"
            ),
            use_container_width=True, hide_index=True
        )
        cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
        st.caption("Confusion matrix · rows = actual, columns = predicted")
        st.dataframe(
            pd.DataFrame(cm, index=["Actual Stay", "Actual Churn"],
                         columns=["Predicted Stay", "Predicted Churn"]),
            use_container_width=True
        )

# ---------------------------------------------------------
# CUSTOMER RISK TABLE
# ---------------------------------------------------------
st.markdown("---")
with st.container(border=True):
    st.markdown('<div class="section-title">CUSTOMER RISK BREAKDOWN TABLE</div>', unsafe_allow_html=True)
    table = pd.DataFrame(index=X.index)
    if "CustomerId" in raw.columns:
        table["Customer ID"] = raw.loc[X.index, "CustomerId"].astype(str)
    else:
        table["Customer ID"] = [f"CUST-{i+1:05d}" for i in range(len(table))]
    if "Surname" in raw.columns:
        table["Customer Name"] = raw.loc[X.index, "Surname"].astype(str)
    table["Risk Score"] = risk["Risk Score"]
    table["Churn Probability"] = risk["Churn Probability"]
    for col in ["Age", "Balance", "Tenure", "Num_Of_Products", "Geography", "Is_Active_Member"]:
        if col in raw.columns:
            table[col.replace("_", " ")] = raw.loc[X.index, col]
    table["Risk Level"] = risk["Risk Level"]
    table["Actual Churn"] = y.map({0: "No", 1: "Yes"})
    table = table.sort_values("Churn Probability", ascending=False)

    search = st.text_input("Search customer ID or name", placeholder="Type to filter the customer table...")
    if search:
        mask = table.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        table = table[mask]

    display_table = table.copy()
    display_table["Churn Probability"] = display_table["Churn Probability"].map(lambda v: f"{v:.1%}")
    st.dataframe(display_table.head(250), use_container_width=True, hide_index=True, height=360)
    st.caption(f"Showing {min(len(display_table), 250):,} of {len(display_table):,} matching customers. Sorted by highest predicted churn probability.")
    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button("Download risk scoring CSV", csv, "customer_churn_risk_scores.csv", "text/csv")

# ---------------------------------------------------------
# INDIVIDUAL CUSTOMER PREDICTION
# ---------------------------------------------------------
st.markdown("---")
st.markdown('<div class="section-title">INDIVIDUAL CUSTOMER RISK ASSESSMENT</div>', unsafe_allow_html=True)
st.caption("Enter a customer's profile to estimate churn probability. Inputs are validated against the training feature ranges.")

with st.container(border=True):
    defaults = X.median(numeric_only=True)
    form_cols = st.columns(3)
    customer_input = {}
    for i, col in enumerate(X.columns):
        target_col = form_cols[i % 3]
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            lo = float(series.min())
            hi = float(series.max())
            default = float(series.median()) if pd.notna(series.median()) else 0.0
            if lo == hi:
                target_col.number_input(col.replace("_", " "), value=default, key=f"cust_{col}")
                customer_input[col] = default
            else:
                customer_input[col] = target_col.number_input(
                    col.replace("_", " "), min_value=lo, max_value=hi,
                    value=min(max(default, lo), hi), key=f"cust_{col}"
                )
        else:
            options = sorted(series.dropna().astype(str).unique().tolist())
            default = str(series.mode(dropna=True).iloc[0]) if not series.mode(dropna=True).empty else (options[0] if options else "")
            customer_input[col] = target_col.selectbox(
                col.replace("_", " "), options=options, index=options.index(default) if default in options else 0,
                key=f"cust_{col}"
            ) if options else ""

    if st.button("Calculate Customer Churn Risk", type="primary"):
        customer_df = pd.DataFrame([customer_input], columns=X.columns)
        customer_prob = float(model.predict_proba(customer_df)[0, 1])
        customer_prediction = int(customer_prob >= 0.5)
        band = risk_band(customer_prob)
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{customer_prob:.1%}")
        c2.metric("Risk Score", f"{customer_prob*100:.0f}/100")
        c3.metric("Risk Segment", band)
        if customer_prediction:
            st.error("Higher likelihood of churn. Consider proactive retention outreach.")
        else:
            st.success("Lower predicted likelihood of churn. Continue regular engagement.")

# ---------------------------------------------------------
# WHAT-IF SIMULATOR
# ---------------------------------------------------------
with st.expander("WHAT-IF SCENARIO SIMULATOR", expanded=False):
    selected_feature = st.selectbox(
        "Choose a numeric feature to adjust",
        X.select_dtypes(include=np.number).columns.tolist()
    )
    if selected_feature:
        feature_min = float(X[selected_feature].min())
        feature_max = float(X[selected_feature].max())
        baseline = float(X[selected_feature].median())

        # Streamlit sliders require min_value < max_value.
        # Skip the slider when a feature is constant across the dataset.
        if np.isclose(feature_min, feature_max):
            st.info(
                f"{selected_feature} has the same value ({feature_min:g}) "
                "for every customer, so it cannot be adjusted in this dataset. "
                "Choose another feature to run a What-If scenario."
            )
        else:
            adjusted = st.slider(
                f"Adjust {selected_feature}",
                min_value=feature_min,
                max_value=feature_max,
                value=min(max(baseline, feature_min), feature_max)
            )
            scenario = X.iloc[[0]].copy()
            scenario[selected_feature] = adjusted
            baseline_prob = float(model.predict_proba(X.iloc[[0]])[0, 1])
            scenario_prob = float(model.predict_proba(scenario)[0, 1])
            a, b, c = st.columns(3)
            a.metric("Baseline probability", f"{baseline_prob:.1%}")
            b.metric("Scenario probability", f"{scenario_prob:.1%}")
            c.metric("Change", f"{(scenario_prob-baseline_prob)*100:+.1f} pp")

# ---------------------------------------------------------
# SHAP EXPLAINABILITY
# ---------------------------------------------------------
with st.expander("SHAP EXPLAINABILITY", expanded=False):
    st.write("SHAP estimates how each input feature contributes to a model prediction. The computation uses a sample to keep the dashboard responsive.")
    shap_sample_size = st.slider("SHAP sample size", 30, min(300, len(X_test)), min(100, len(X_test)))
    if st.button("Generate SHAP global explanation"):
        try:
            import shap
            with st.spinner("Calculating SHAP values..."):
                prep = model.named_steps["prep"]
                estimator = model.named_steps["model"]
                sample = X_test.sample(shap_sample_size, random_state=42)
                transformed = prep.transform(sample)
                names = get_feature_names(model)
                if hasattr(transformed, "toarray"):
                    transformed = transformed.toarray()

                if isinstance(estimator, (RandomForestClassifier, GradientBoostingClassifier)):
                    explainer = shap.TreeExplainer(estimator)
                    values = explainer.shap_values(transformed)
                else:
                    background = prep.transform(X_train.sample(min(100, len(X_train)), random_state=42))
                    if hasattr(background, "toarray"):
                        background = background.toarray()
                    explainer = shap.LinearExplainer(estimator, background)
                    values = explainer.shap_values(transformed)

                if isinstance(values, list):
                    values = values[1] if len(values) > 1 else values[0]
                values = np.asarray(values)
                if values.ndim == 3:
                    values = values[:, :, 1]
                if values.ndim != 2:
                    raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

                mean_abs = np.abs(values).mean(axis=0)
                shap_importance = pd.DataFrame({
                    "Feature": names,
                    "Mean |SHAP value|": mean_abs
                }).sort_values("Mean |SHAP value|", ascending=False).head(15)

                fig, ax = plt.subplots(figsize=(8, 5))
                plot_data = shap_importance.sort_values("Mean |SHAP value|")
                ax.barh(plot_data["Feature"], plot_data["Mean |SHAP value|"], color="#ef793d")
                ax.set_title("Global SHAP feature importance")
                ax.set_xlabel("Mean absolute SHAP value")
                st.pyplot(styled_chart(fig), use_container_width=True)
                plt.close(fig)
                st.dataframe(shap_importance, use_container_width=True, hide_index=True)
        except ImportError:
            st.error("SHAP is not installed. Run: pip install shap")
        except Exception as exc:
            st.error(f"SHAP could not be generated for this model/version: {exc}")

st.markdown("---")
st.markdown(
    '<div class="subtle">QUANTUM · Customer churn risk scoring is a decision-support estimate, not a guarantee of customer behaviour. Validate model performance and thresholds before operational use.</div>',
    unsafe_allow_html=True
)
