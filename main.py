import streamlit as st
import pandas as pd
import joblib

# Function to load the trained model
def load_model():
    model = joblib.load('decision_tree.pkl')
    return model

# Function to preprocess data and make predictions
def predict(model, user_input):
    user_df = pd.DataFrame(user_input, index=[0])
    
    # Preprocess user input to match model training data format
    user_df['Dependents_0'] = user_df['Dependents'] == 0
    user_df['Dependents_1'] = user_df['Dependents'] == 1
    user_df['Dependents_2'] = user_df['Dependents'] == 2
    user_df['Dependents_3+'] = user_df['Dependents'] >= 3
    
    # Drop the original Dependents column
    user_df = user_df.drop(columns=['Dependents'])
    
    # Ensure all columns are in the correct order and missing columns are filled with False
    expected_columns = [
        'Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome',
        'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
        'Property_Area', 'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+'
    ]
    
    for col in expected_columns:
        if col not in user_df.columns:
            user_df[col] = False
    
    user_df = user_df[expected_columns]
    
    # Make prediction
    prediction = model.predict(user_df)
    
    return prediction[0]

# Streamlit UI
def main():
    st.title('Loan Approval Prediction')
    
    background_image_url = 'https://www.infomerics.com/admin/uploads/1612367957banner2.jpg'  # Replace with your image URL
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
    # Load model
    model = load_model()
    
    # User input form in the main body
    st.header('User Input Features')
    
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Marital Status', ['Yes', 'No'])
    dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, step=1)
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    applicant_income = st.number_input('Applicant Income', min_value=0)
    coapplicant_income = st.number_input('Coapplicant Income', min_value=0)
    loan_amount = st.number_input('Loan Amount', min_value=0)
    loan_amount_term = st.number_input('Loan Amount Term', min_value=0)
    credit_history = st.selectbox('Credit History', [0, 1])
    property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])
    
    user_input = {
        'Gender': 1 if gender == 'Male' else 0,
        'Married': 1 if married == 'Yes' else 0,
        'Dependents': dependents,
        'Education': 1 if education == 'Graduate' else 0,
        'Self_Employed': 1 if self_employed == 'Yes' else 0,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': 2 if property_area == 'Urban' else (1 if property_area == 'Semiurban' else 0)
    }
    
    # Make prediction
    if st.button('Predict'):
        try:
            prediction = predict(model, user_input)
            if prediction == 0:
                st.success('Congratulations! Your loan is approved.')
            else:
                st.error('Sorry, your loan is not approved.')
        except ValueError as e:
            st.error(str(e))

if __name__ == '__main__':
    main()
