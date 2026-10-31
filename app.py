
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the trained model
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "deployment" / "best_model.joblib"

model = joblib.load(MODEL_PATH)

# Streamlit application title
st.title("Wellness Tourism Purchase Prediction")

st.write(
    "This application predicts whether a customer is likely "
    "to purchase the Wellness Tourism package based on their profile."
)

st.write("Please enter the customer details below.")

# Customer categorical inputs

st.subheader("Customer Profile")

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)

Gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Fe Male"]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Divorced", "Married", "Unmarried"]
)

Designation = st.selectbox(
    "Designation",
    ["Manager", "Executive", "Senior Manager", "AVP", "VP"]
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"]
)

# Customer numerical inputs

st.subheader("Customer Details")

Age = st.number_input(
    "Age", min_value=18, max_value=100, value=30
)

CityTier = st.selectbox(
    "City Tier", [1, 2, 3]
)

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting", min_value=1, max_value=10, value=2
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star", [3, 4, 5]
)

NumberOfTrips = st.number_input(
    "Number of Trips", min_value=0, max_value=100, value=3
)

Passport = st.selectbox(
    "Does the Customer Have a Passport?", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

OwnCar = st.selectbox(
    "Does the Customer Own a Car?", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting", min_value=0, max_value=10, value=0
)

MonthlyIncome = st.number_input(
    "Monthly Income (₹)", min_value=0, value=25000
)

PitchSatisfactionScore = st.selectbox(
    "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
)

NumberOfFollowups = st.number_input(
    "Number of Follow-ups", min_value=0, max_value=10, value=3
)

DurationOfPitch = st.number_input(
    "Duration of Pitch (minutes)", min_value=0, value=15
)

# Create input DataFrame

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}])

classification_threshold = 0.45

# Prediction button
if st.button("Predict Purchase"):

    prediction_proba = model.predict_proba(input_data)[0, 1]

    prediction = int(prediction_proba >= classification_threshold)

    if prediction == 1:
        st.success("The customer is likely to purchase the Wellness Tourism package.")
    else:
        st.warning("The customer is unlikely to purchase the Wellness Tourism package.")

    st.write(f"Purchase Probability: {prediction_proba:.2%}")
