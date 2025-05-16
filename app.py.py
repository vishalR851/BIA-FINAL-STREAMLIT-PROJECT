import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle

st.set_page_config(page_title="Customer Churn Prediction App", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("churn.csv")
    df = df.drop(['customerID'], axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    cat_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    df_encoded = df.copy()

    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, df_encoded, label_encoders

# Main app
def main():
    df, df_encoded, label_encoders = load_data()

    st.title("📊 Customer Churn Prediction App")

    menu = ["Home", "Exploratory Analysis", "Model Training", "Predict Churn"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.markdown("### 📄 Dataset Preview")
        st.dataframe(df.head())
        st.markdown("### 📈 Churn Distribution")
        churn_count = df['Churn'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(churn_count, labels=churn_count.index, autopct='%1.1f%%', startangle=90, colors=["lightgreen", "lightcoral"])
        ax.axis('equal')
        st.pyplot(fig)

    elif choice == "Exploratory Analysis":
        st.markdown("## 📊 Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔍 Numerical Summary")
            st.dataframe(df.describe())
        with col2:
            st.markdown("### 🔍 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df_encoded.corr(), cmap='coolwarm', annot=True, fmt=".2f", ax=ax)
            st.pyplot(fig)

        categorical_cols = df.select_dtypes(include=['object']).columns
        if 'Churn' in categorical_cols:
            categorical_cols = categorical_cols.drop('Churn')

        selected_cat = st.selectbox("Select a categorical column to visualize", categorical_cols)
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=selected_cat, hue='Churn', ax=ax)
        st.pyplot(fig)

    elif choice == "Model Training":
        st.markdown("## 🧠 Train Your Model")

        X = df_encoded.drop('Churn', axis=1)
        y = df_encoded['Churn']

        use_smote = st.checkbox("Apply SMOTE to handle class imbalance?", value=True)
        if use_smote:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_choice = st.selectbox("Choose a Model", ["Random Forest", "Logistic Regression", "Support Vector Machine"])
        if model_choice == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif model_choice == "Logistic Regression":
            model = LogisticRegression()
        else:
            model = SVC(probability=True)

        if st.button("Train Model"):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            st.subheader("✅ Model Evaluation")
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
            st.text("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))
            st.text("Classification Report")
            st.text(classification_report(y_test, y_pred))

            # Save model and scaler
            with open("trained_model.pkl", "wb") as f:
                pickle.dump(model, f)
            with open("scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            with open("label_encoders.pkl", "wb") as f:
                pickle.dump(label_encoders, f)

            st.success("Model and Scaler saved successfully.")
            st.download_button("📥 Download Model", data=open("trained_model.pkl", "rb"), file_name="trained_model.pkl")

    elif choice == "Predict Churn":
        st.markdown("## 🔮 Predict Customer Churn")

        try:
            with open("trained_model.pkl", "rb") as f:
                model = pickle.load(f)
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            with open("label_encoders.pkl", "rb") as f:
                label_encoders = pickle.load(f)
        except FileNotFoundError:
            st.error("Please train the model first in the 'Model Training' section.")
            return

        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with col2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
            total_charges = st.number_input("Total Charges", min_value=0.0, step=1.0)
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

        input_dict = {
            'gender': gender,
            'SeniorCitizen': 1 if senior == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple,
            'InternetService': internet,
            'OnlineSecurity': online_security,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        input_df = pd.DataFrame([input_dict])
        for col in input_df.select_dtypes(include='object').columns:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])

        input_scaled = scaler.transform(input_df)

        if st.button("Predict Churn"):
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            st.subheader("📢 Prediction Result")
            st.write(f"**Churn Prediction:** {'Yes' if prediction == 1 else 'No'}")
            st.write(f"**Churn Probability:** {probability:.2%}")

if __name__ == "__main__":
    main()
