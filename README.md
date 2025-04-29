# Titanic Survival Predictor

This project uses machine learning to predict the survival probability of Titanic passengers based on their characteristics.

## Setup

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Local Development

1. Train the model:

```bash
python titanic_ml.py
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

## Deployment

This app can be deployed on Streamlit Cloud. Follow these steps:

1. Create a GitHub repository and push this code
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository
6. Set the main file path to `app.py`
7. Click "Deploy"

## Project Structure

- `titanic_ml.py`: Contains the machine learning model and data preprocessing
- `app.py`: Streamlit web application for making predictions
- `requirements.txt`: Project dependencies
- `data/`: Directory for the Titanic dataset
- `models/`: Directory for saved models

## Features

- Machine learning model using Random Forest Classifier
- Data preprocessing and feature engineering
- Interactive web interface
- Real-time predictions
- Visualizations of historical survival rates

## Model Features

The model uses the following features to make predictions:

- Passenger Class
- Sex
- Age
- Number of Siblings/Spouses
- Number of Parents/Children
- Fare
- Port of Embarkation
- Derived features (Family Size, Is Alone, Title)
