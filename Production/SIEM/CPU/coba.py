import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Load your pre-trained BiLSTM model and scaler
model_bilstm = tf.keras.models.load_model("my_model-bilstm-window_12.keras")
scaler = joblib.load("scaler.pkl")

# Create FastAPI instance
app = FastAPI(title="Deep Learning Forecast API")


def getData(step):
    end_time = int(time.time())
    start_time = end_time - (60 * 60 * 6)  # last 2 hours of data
    # 'step' is passed as parameter (in seconds)
    prometheus_url = "http://203.175.10.196:9090/api/v1/query_range"
    query = '100 * (1 - avg by(job)(irate(node_cpu_seconds_total{job=~"SIEM", mode="idle"}[1m])))'
    params = {"query": query, "start": start_time, "end": end_time, "step": step}

    response = requests.get(prometheus_url, params=params)
    data = response.json()
    try:
        result = data["data"]["result"][0]["values"]
    except (KeyError, IndexError):
        raise ValueError("Prometheus query did not return any data.")

    # Extract the last 4 data points (since the model was trained with a window of 4)
    window_values = [float(point[1]) for point in result[-20:]]

    # Convert to NumPy array and reshape to (4, 1) as required by scaler.transform
    window_values = np.array(window_values).reshape(-1, 1)
    # Scale the values (model expects scaled inputs)
    window_values_scaled = scaler.transform(window_values)
    # Reshape to (1, 4, 1) to match the model input shape
    seed = window_values_scaled.reshape(1, -1, 1)
    return seed


# For a direct one-step forecast (no recursion needed)
def direct_forecast(model, seed):
    # Predict one step ahead using the seed data
    pred = model.predict(seed)  # Expected shape: (1, 1)
    # Inverse-transform the prediction to get the original scale.
    pred_original = scaler.inverse_transform(pred)
    return float(pred_original[0, 0])


@app.get("/")
def read_root():
    return {"message": "Welcome to the LSTM Forecasting API!"}


# Example usage in your FastAPI endpoint
@app.get("/test_forecast")
def test_forecast():
    # For example, pass a step interval of 1800 seconds (adjust as needed)
    seed = getData(1800)
    # print("Seed :", seed)

    forecast_value = direct_forecast(model_bilstm, seed)
    # print("Forecast (next step):", forecast_value)

    return {
        "message": "Test Forecast Successful",
        "Prediction (next step)": forecast_value,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
