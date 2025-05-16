import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import pickle

# Set Streamlit page configuration
st.set_page_config(
    page_title="Customer Churn Prediction", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://example.com/help',
        'Report a bug': "https://example.com/bug",
        'About': "# Customer Churn Prediction Dashboard"
    }
)

# Custom CSS for modern look
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .stButton>button {border-radius: 8px; background-color: #4CAF50; color: white; border: none; padding: 8px 16px;}
        .stSelectbox>div>div {background-color: #f0f2f6; border-radius: 8px;}
        .stSlider>div {color: #0c4a6e;}
        .css-1d391kg {background-color: #ffffff; border-radius: 8px; padding: 1rem;}
        .stAlert {border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Customer Churn Prediction Dashboard")

# Initialize session state for model persistence
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None

# Sidebar navigation
with st.sidebar:
    st.title("🔍 Navigation")
    options = st.radio("Go to", ["Data Overview", "Exploratory Analysis", "Model Training", "Prediction"])
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This dashboard helps predict customer churn using machine learning.")

# Load data
@st.cache_data
def load_data():
    try:
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        data = pd.read_csv(url)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_data()

if df is None:
    st.stop()

# Preprocessing function
@st.cache_data
def preprocess_data(df):
    # Make a copy to avoid modifying cached data
    df = df.copy()
    
    # Drop customer ID
    df = df.drop('customerID', axis=1)
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Encode target
    df['Churn'] = LabelEncoder().fit_transform(df['Churn'])
    
    # Get categorical columns (excluding target)
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df, df_encoded

df, df_encoded = preprocess_data(df)

# Page: Data Overview
if options == "Data Overview":
    st.header("📁 Data Overview")

    with st.expander("🔍 View Dataset", expanded=True):
        st.dataframe(df, use_container_width=True)

    st.subheader("📐 Dataset Shape")
    st.write(f"*Rows:* {df.shape[0]} | *Columns:* {df.shape[1]}")

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🧾 Column Data Types"):
            st.write(df.dtypes)
    
    with col2:
        with st.expander("🚨 Missing Values"):
            st.write(df.isnull().sum())

# Page: Exploratory Analysis
elif options == "Exploratory Analysis":
    st.header("📊 Exploratory Data Analysis")

    st.subheader("Churn Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x='Churn', data=df, palette='Set2', ax=ax)
    ax.set_title('Customer Churn Distribution')
    st.pyplot(fig)

    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numeric Features")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        selected_num_col = st.selectbox("📈 Select a numeric feature to visualize", numeric_cols)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[selected_num_col], kde=True, color="skyblue", ax=ax)
        ax.set_title(f'Distribution of {selected_num_col}')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Categorical Features")
        categorical_cols = df.select_dtypes(include=['object']).columns.drop('Churn')
        selected_cat_col = st.selectbox("🗂 Select a categorical feature to visualize", categorical_cols)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=selected_cat_col, hue='Churn', data=df, palette='Set1', ax=ax)
        plt.xticks(rotation=45)
        ax.set_title(f'Churn by {selected_cat_col}')
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("📌 Correlation Matrix")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Feature Correlation Matrix')
    st.pyplot(fig)

# Page: Model Training
elif options == "Model Training":
    st.header("🧠 Model Training")
    
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    st.sidebar.subheader("🔧 Model Configuration")
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random State", 0, 100, 42)
    use_smote = st.sidebar.checkbox("Balance Classes with SMOTE")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    
    st.subheader("Select and Train Model")
    model_type = st.selectbox("Choose Model", 
                            ["Random Forest", "Logistic Regression", "Support Vector Machine"])
    
    if model_type == "Random Forest":
        n_estimators = st.slider("Number of Trees", 50, 300, 100, 25)
        max_depth = st.slider("Max Depth", 2, 20, 6)
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )
    elif model_type == "Logistic Regression":
        C = st.slider("C (Regularization Strength)", 0.01, 10.0, 1.0)
        model = LogisticRegression(
            C=C, 
            max_iter=1000, 
            random_state=random_state
        )
    else:
        C = st.slider("C (SVM Regularization)", 0.01, 10.0, 1.0)
        kernel = st.selectbox("SVM Kernel", ["linear", "rbf", "poly"])
        model = SVC(
            C=C, 
            kernel=kernel, 
            probability=True, 
            random_state=random_state
        )
    
    if st.button("🚀 Train Model"):
        with st.spinner('Training model...'):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            # Store model and artifacts in session state
            st.session_state.trained_model = model
            st.session_state.feature_names = X.columns.tolist()
            st.session_state.preprocessor = {
                'scaler': scaler,
                'use_smote': use_smote
            }
            
            st.success("Model trained successfully!")
            
            # Evaluation metrics
            st.subheader("📈 Evaluation Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
            with col2:
                st.metric("Positive Class Rate", f"{y_test.mean():.2%}")
            
            st.markdown("---")
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(
                    confusion_matrix(y_test, y_pred), 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    ax=ax
                )
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
            
            with col2:
                st.subheader("ROC Curve")
                from sklearn.metrics import RocCurveDisplay
                fig, ax = plt.subplots()
                RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax)
                ax.plot([0, 1], [0, 1], linestyle='--')
                st.pyplot(fig)
            
            if model_type == "Random Forest":
                st.markdown("---")
                st.subheader("Feature Importance")
                importance = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": model.feature_importances_
                }).sort_values("Importance", ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    x="Importance", 
                    y="Feature", 
                    data=importance.head(10),
                    ax=ax
                )
                ax.set_title('Top 10 Important Features')
                st.pyplot(fig)
                
                # Option to download feature importance
                st.download_button(
                    label="Download Feature Importance",
                    data=importance.to_csv(index=False),
                    file_name="feature_importance.csv",
                    mime="text/csv"
                )

# Page: Prediction
elif options == "Prediction":
    st.header("🔮 Predict Customer Churn")
    
    if st.session_state.trained_model is None:
        st.warning("Please train a model first on the Model Training page.")
        st.stop()
    
    st.subheader("📝 Customer Details")
    
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)
        senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", 
                                      "Bank transfer", "Credit card"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    if st.button("🔍 Predict Churn"):
        try:
            # Create input DataFrame
            input_data = pd.DataFrame({
                'tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
                'SeniorCitizen_Yes': [1 if senior_citizen == "Yes" else 0],
                'Partner_Yes': [1 if partner == "Yes" else 0],
                'Dependents_Yes': [1 if dependents == "Yes" else 0],
                'Gender_Male': [1 if gender == "Male" else 0],
                'Contract_One year': [1 if contract == "One year" else 0],
                'Contract_Two year': [1 if contract == "Two year" else 0],
                'InternetService_Fiber optic': [1 if internet_service == "Fiber optic" else 0],
                'InternetService_No': [1 if internet_service == "No" else 0],
                'PaymentMethod_Credit card (automatic)': [1 if payment_method == "Credit card" else 0],
                'PaymentMethod_Electronic check': [1 if payment_method == "Electronic check" else 0],
                'PaymentMethod_Mailed check': [1 if payment_method == "Mailed check" else 0],
                'PhoneService_Yes': [1 if phone_service == "Yes" else 0],
                'MultipleLines_No phone service': [1 if multiple_lines == "No phone service" else 0],
                'MultipleLines_Yes': [1 if multiple_lines == "Yes" else 0]
            })
            
            # Ensure all expected features are present
            missing_features = set(st.session_state.feature_names) - set(input_data.columns)
            for feature in missing_features:
                input_data[feature] = 0  # Add missing features with 0 value
            
            # Reorder columns to match training data
            input_data = input_data[st.session_state.feature_names]
            
            # Scale features
            scaler = st.session_state.preprocessor['scaler']
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            model = st.session_state.trained_model
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Churn", "Yes" if prediction[0] == 1 else "No")
            with col2:
                st.metric("Probability", f"{probability:.2%}")
            
            # Show explanation
            if prediction[0] == 1:
                st.warning("This customer is likely to churn.")
            else:
                st.success("This customer is likely to stay.")
            
            # Show feature importance if available
            if hasattr(model, 'feature_importances_'):
                st.markdown("---")
                st.subheader("Top Factors Influencing Prediction")
                
                # Get feature importances
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:5]  # Top 5 features
                
                # Create explanation
                explanation = []
                for i in indices:
                    feature_name = st.session_state.feature_names[i]
                    feature_value = input_data.iloc[0][feature_name]
                    importance = importances[i]
                    
                    # Create human-readable explanation
                    if feature_name.startswith("Contract_Two year") and feature_value == 1:
                        explanation.append(f"✅ Two year contract (reduces churn)")
                    elif feature_name == "tenure":
                        explanation.append(f"✅ Longer tenure ({tenure} months)")
                    elif feature_name == "MonthlyCharges" and monthly_charges > 70:
                        explanation.append(f"⚠ High monthly charges (${monthly_charges})")
                    
                if explanation:
                    st.write("Key factors:")
                    for item in explanation:
                        st.write(f"- {item}")
                else:
                    st.info("No strong factors identified.")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
