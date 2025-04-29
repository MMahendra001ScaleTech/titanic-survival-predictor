import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_data():
    # Load the Titanic dataset
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    return train_data, test_data

def preprocess_data(df):
    # Create a copy of the dataframe
    df = df.copy()
    
    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Create new features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Extract title from name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    return df

def train_model():
    # Load and preprocess data
    train_data, _ = load_data()
    train_data = preprocess_data(train_data)
    
    # Define features and target
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title']
    X = train_data[features]
    y = train_data['Survived']
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing pipeline
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Create and train the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, 'models/titanic_model.joblib')
    
    return model

def predict_survival(model, passenger_data):
    # Preprocess the input data
    passenger_df = pd.DataFrame([passenger_data])
    passenger_df = preprocess_data(passenger_df)
    
    # Make prediction
    prediction = model.predict(passenger_df)
    probability = model.predict_proba(passenger_df)
    
    return prediction[0], probability[0][1]

if __name__ == "__main__":
    # Train the model
    model = train_model()
    
    # Example prediction
    example_passenger = {
        'Pclass': 1,
        'Sex': 'female',
        'Age': 25,
        'SibSp': 0,
        'Parch': 0,
        'Fare': 50,
        'Embarked': 'S',
        'Name': 'Example, Miss. Test'
    }
    
    prediction, probability = predict_survival(model, example_passenger)
    print(f"\nExample Prediction:")
    print(f"Survival Prediction: {'Survived' if prediction == 1 else 'Did not survive'}")
    print(f"Survival Probability: {probability:.2%}") 