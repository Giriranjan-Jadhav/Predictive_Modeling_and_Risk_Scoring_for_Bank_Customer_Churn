import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="Churn Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel(r"C:\Users\GIRIRANJAN\OneDrive\Desktop\European_Bank.xlsx")
    return df

df = load_data()

@st.cache_data
def preprocess(df):
    df = df.copy()

    df['Balance_Salary_Ratio'] = df['Balance'] / df['Estimated_Salary']
    df['Product_Density'] = df['Num_Of_Products'] / df['Tenure']
    df['Engagement_Product'] = df['Is_Active_Member'] * df['Num_Of_Products']
    df['Age_Tenure'] = df['Age'] * df['Tenure']

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df

df = preprocess(df)

from sklearn.model_selection import train_test_split

X = df.drop(['Exited', 'Surname'], axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

@st.cache_resource
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

trained_models = train_models(X_train, y_train)

rf = trained_models["Random Forest"]
lr = trained_models["Logistic Regression"]
gr = trained_models["Gradient Boosting"]

st.sidebar.header("Enter Customer Details")

input_data = {}
for col in X.columns:
    input_data[col] = st.sidebar.number_input(
        f"{col}",
        float(X[col].min()),
        float(X[col].max()),
        float(X[col].mean())
    )

input_df = pd.DataFrame([input_data])

model_choice = st.selectbox("Choose Model", ["Random Forest", "Logistic Regression","Gradient Boosting"])

model = trained_models[model_choice]

st.title("Customer Churn Prediction Dashboard")

prob = model.predict_proba(input_df)[0][1]
prediction = model.predict(input_df)[0]

st.subheader("Prediction Result")

st.write(f"**Churn Probability:** {prob:.2f}")

if prediction == 1:
    st.error("Customer Likely to Churn")
else:
    st.success("Customer Likely to Stay")

st.subheader("Probability Distribution")

probs = model.predict_proba(X_test)[:, 1]

fig, ax = plt.subplots()
ax.hist(probs, bins=20)
ax.set_title("Churn Probability Distribution")
st.pyplot(fig)

st.subheader("Feature Importance")

if model_choice in ["Random Forest", "Gradient Boosting"]:
    importances = model.feature_importances_
else:
    importances = np.abs(model.coef_[0])

feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.bar_chart(feat_imp.set_index("Feature"))

st.subheader("What-If Scenario Simulator")

selected_feature = st.selectbox("Select Feature", X.columns)

min_val = float(X[selected_feature].min())
max_val = float(X[selected_feature].max())

if min_val == max_val:
    min_val -= 1
    max_val += 1

new_value = st.slider(
    f"Adjust {selected_feature}",
    min_val,
    max_val,
    float(input_df[selected_feature].values[0])
)

temp_df = input_df.copy()
temp_df[selected_feature] = new_value

new_prob = model.predict_proba(temp_df)[0][1]

st.write(f"New Churn Probability: {new_prob:.2f}")
input_df[selected_feature] = new_value

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.subheader("Model Evaluation")

y_pred = model.predict(X_test)

st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall:", recall_score(y_test, y_pred))
st.write("F1 Score:", f1_score(y_test, y_pred))

st.subheader("SHAP Explainability")

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:50], check_additivity=False)

fig, ax = plt.subplots()
shap.plots.beeswarm(shap_values[:, :, 1], show=False)
st.pyplot(fig)

@st.cache_resource
def compute_shap(_model, X_train, X_test):
    explainer = shap.Explainer(_model, X_train)
    shap_values = explainer(X_test[:200], check_additivity=False)
    return shap_values

if st.button("Generate SHAP Plots"):
    with st.spinner("Generating SHAP... ⏳"):

        shap_values = compute_shap(model, X_train, X_test)

        st.write("### SHAP Summary Plot")
        fig1 = plt.figure()
        import shap
        shap.summary_plot(shap_values[:200], X_test[:200], show=False)
        st.pyplot(fig1)

        st.write("### SHAP Feature Importance")
        fig2 = plt.figure()
        shap.summary_plot(shap_values[:200], X_test[:200], plot_type="bar", show=False)
        st.pyplot(fig2)
