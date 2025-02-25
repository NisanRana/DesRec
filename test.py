from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import requests
import base64
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
@st.cache_data
def load_data():
    data = pd.read_csv("destinations_with_coordinates.csv")

    # Data Cleaning & Preprocessing
    data.columns = data.columns.str.strip().str.lower()
    data["pname"] = data["pname"].str.strip().str.lower()
    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")

    feature_cols = ['culture', 'adventure', 'wildlife', 'sightseeing', 'history']
    scaler = MinMaxScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])

    # Target Variable (Popularity)
    data['popularity'] = data[feature_cols].mean(axis=1) + 0.1 * data['culture']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        data[feature_cols], data['popularity'], test_size=0.2, random_state=42
    )

    # Train LightGBM Model
    lgbm_model = lgbm.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=100)
    lgbm_model.fit(X_train, y_train)

    # Predictions
    y_pred = lgbm_model.predict(X_test)

    # Calculate Accuracy Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.sidebar.subheader("📊 Model Performance")
    st.sidebar.text(f"MAE: {mae:.4f}")
    st.sidebar.text(f"MSE: {mse:.4f}")
    st.sidebar.text(f"RMSE: {rmse:.4f}")
    st.sidebar.text(f"R² Score: {r2:.4f}")

    # Store ML Score
    data['ml_score'] = lgbm_model.predict(data[feature_cols])
    data['ml_score'] = MinMaxScaler().fit_transform(data[['ml_score']])

    return data, scaler, feature_cols, lgbm_model

data, scaler, feature_cols, lgbm_model = load_data()
