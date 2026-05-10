# Used Cars Price Prediction API

A machine learning pipeline to predict used car prices using multiple regression models, deployed via FastAPI.

## Features

- **Data Processing**: Extracts horsepower and liters from engine column, handles missing values, caps outliers
- **Models**: Linear Regression, Decision Tree, Logistic Regression (price categorization)
- **API**: FastAPI endpoints for real-time predictions
- **Encoding**: One-hot encoding for low-cardinality features, label encoding for high-cardinality

## Setup

```bash
pip install -r requirements.txt
python app.py