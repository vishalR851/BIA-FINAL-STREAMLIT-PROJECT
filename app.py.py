import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Customer Churn Prediction App", layout="wide")

# Title
st.title("📊 Customer Churn Prediction App")

# Upload Dataset
st.subheader("Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.dataframe(df.head())
else:
    st.warning("Please upload a dataset to proceed.")
    st.stop()

# Preprocessing
def preprocess_data(df):
    df = df.copy()

    # Drop customerID if present
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric (handle missing values)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Encode target
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Label encode binary columns
    binary_cols = df.nunique()[df.nunique() == 2].keys().tolist()
    if 'Churn' in binary_cols:
        binary_cols.remove('Churn')
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    # One-hot encode remaining categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Feature/target split
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle imbalance
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X.columns

# Sidebar for model selection
st.sidebar.header("Choose a Model")
model_choice = st.sidebar.selectbox("Select Classifier", ["Logistic Regression", "Random Forest", "SVM"])

# Train & Predict
if st.sidebar.button("Train Model"):
    st.subheader("📈 Model Training and Evaluation")

    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)

    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = SVC(probability=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.metric("Accuracy", f"{acc:.2f}")
    st.text("Confusion Matrix:")
    st.write(cm)

    st.text("Classification Report:")
    st.json(report)

    # Save model
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # SHAP Explainability
    st.subheader("🔍 Model Explainability with SHAP")

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test[:100])

    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.title("Feature Importance")
    shap.plots.bar(shap_values)
    st.pyplot(bbox_inches='tight')

    # Force plot
    st.subheader("SHAP Force Plot (First Prediction)")
    shap.initjs()
    shap.plots.force(shap_values[0])
    st.pyplot(bbox_inches='tight')
