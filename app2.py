import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model (without preprocessing)
model = joblib.load(r"C:\Users\gunav\Desktop\employee_salary_prediction\best_model.pkl")

# Load or define encoders for categorical columns (Replace these with your saved encoders if available)
# For demo, we create and fit LabelEncoders with expected categories (must exactly match training!)
def get_label_encoders():
    encoders = {}

    # Replace the categories with the exact categories from training data
    encoders['workclass'] = LabelEncoder().fit([
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ])

    encoders['marital-status'] = LabelEncoder().fit([
        "Married-civ-spouse", "Divorced", "Never-married",
        "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ])

    encoders['occupation'] = LabelEncoder().fit([
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
    ])

    encoders['relationship'] = LabelEncoder().fit([
        "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
    ])

    encoders['race'] = LabelEncoder().fit([
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
    ])

    encoders['gender'] = LabelEncoder().fit(["Female", "Male"])

    encoders['native-country'] = LabelEncoder().fit([
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
        "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece",
        "South", "China", "Cuba", "Iran", "Honduras", "Philippines",
        "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal",
        "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador",
        "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua",
        "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
        "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"
    ])

    return encoders

encoders = get_label_encoders()

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

# Collect inputs exactly matching training features

age = st.sidebar.number_input("Age", 18, 90, 30)

workclass = st.sidebar.selectbox("Workclass", encoders['workclass'].classes_)

fnlwgt = st.sidebar.number_input("Fnlwgt", 1, 1000000, 100000)

educational_num = st.sidebar.number_input("Educational Number", 1, 16, 10)

marital_status = st.sidebar.selectbox("Marital Status", encoders['marital-status'].classes_)

occupation = st.sidebar.selectbox("Occupation", encoders['occupation'].classes_)

relationship = st.sidebar.selectbox("Relationship", encoders['relationship'].classes_)

race = st.sidebar.selectbox("Race", encoders['race'].classes_)

gender = st.sidebar.selectbox("Gender", encoders['gender'].classes_)

capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)

capital_loss = st.sidebar.number_input("Capital Loss", 0, 100000, 0)

hours_per_week = st.sidebar.number_input("Hours per week", 1, 99, 40)

native_country = st.sidebar.selectbox("Native Country", encoders['native-country'].classes_)

# Create DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

# Encode categorical columns
for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
    input_df[col] = encoders[col].transform(input_df[col])

st.write("### 🔎 Input Data After Encoding")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    pred_label = ">50K" if prediction[0] == 1 else "<=50K"
    st.success(f"✅ Predicted Income: {pred_label}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)

    # Encode batch categorical columns same way
    for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
        if col in batch_df.columns:
            batch_df[col] = batch_df[col].map(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)

    st.write("Uploaded data preview (encoded):")
    st.write(batch_df.head())

    batch_preds = model.predict(batch_df)
    batch_df['PredictedIncome'] = np.where(batch_preds == 1, ">50K", "<=50K")

    st.write("✅ Predictions:")
    st.write(batch_df.head())

    csv = batch_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name="batch_predictions.csv", mime="text/csv")
