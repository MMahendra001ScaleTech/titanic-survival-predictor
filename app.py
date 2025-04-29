import streamlit as st
import joblib
import pandas as pd
from titanic_ml import preprocess_data, predict_survival

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('models/titanic_model.joblib')

def main():
    st.title("Titanic Survival Predictor")
    st.write("""
    This app predicts whether a passenger would have survived the Titanic disaster
    based on their characteristics.
    """)
    
    # Sidebar for user input
    st.sidebar.header('Passenger Information')
    
    # Input fields
    pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3])
    sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    age = st.sidebar.slider('Age', 0, 100, 25)
    sibsp = st.sidebar.slider('Number of Siblings/Spouses', 0, 8, 0)
    parch = st.sidebar.slider('Number of Parents/Children', 0, 6, 0)
    fare = st.sidebar.slider('Fare', 0, 500, 50)
    embarked = st.sidebar.selectbox('Port of Embarkation', ['C', 'Q', 'S'])
    name = st.sidebar.text_input('Name', 'Example, Mr. Test')
    
    # Create passenger data dictionary
    passenger_data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked,
        'Name': name
    }
    
    # Load model and make prediction
    model = load_model()
    
    if st.sidebar.button('Predict'):
        prediction, probability = predict_survival(model, passenger_data)
        
        # Display results
        st.subheader('Prediction Results')
        if prediction == 1:
            st.success(f'Prediction: Survived (Probability: {probability:.2%})')
        else:
            st.error(f'Prediction: Did not survive (Probability: {1-probability:.2%})')
        
        # Display feature importance with proper data types
        st.subheader('Feature Importance')
        feature_data = {
            'Feature': ['Passenger Class', 'Sex', 'Age', 'Siblings/Spouses', 
                       'Parents/Children', 'Fare', 'Embarkation Port'],
            'Value': [str(pclass), sex, str(age), str(sibsp), str(parch), str(fare), embarked]
        }
        feature_importance = pd.DataFrame(feature_data)
        st.dataframe(feature_importance, use_container_width=True)
        
        # Add some visualizations
        st.subheader('Survival Statistics')
        col1, col2 = st.columns(2)
        
        with col1:
            st.write('Passenger Class Distribution')
            class_data = pd.DataFrame({
                'Class': [1, 2, 3],
                'Survival Rate': [0.63, 0.47, 0.24]  # Approximate historical rates
            }).set_index('Class')
            st.bar_chart(class_data)
        
        with col2:
            st.write('Gender Survival Rate')
            gender_data = pd.DataFrame({
                'Gender': ['Female', 'Male'],
                'Survival Rate': [0.74, 0.19]  # Approximate historical rates
            }).set_index('Gender')
            st.bar_chart(gender_data)

if __name__ == '__main__':
    main() 