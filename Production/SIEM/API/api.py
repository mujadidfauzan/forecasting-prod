import time

import joblib
import numpy as np
import requests
import tensorflow as tf
import uvicorn
from fastapi import FastAPI

# ---------------------------
# 1) Load your models & scalers
# ---------------------------
# CPU model + scaler
model_bilstm_cpu = tf.keras.models.load_model("my_model-bilstm-window_12.keras")
scaler_cpu = joblib.load("scaler-cpu.pkl")  # Rename your CPU scaler if they differ

# Memory model + scaler
model_bilstm_mem = tf.keras.models.load_model("my_model-bilstm-window_4.keras")
scaler_mem = joblib.load("scaler-memo.pkl")  # Rename your MEM scaler if they differ

# ---------------------------
# 2) Create FastAPI instance
# ---------------------------
app = FastAPI(title="Deep Learning Forecast API")


# ---------------------------
# 3) Utility functions
# ---------------------------
def get_cpu_data(step: int):
    """
    Retrieves recent Prometheus CPU usage data for 'SIEM'.
    Returns the scaled seed array matching the CPU model input shape.
    """
    end_time = int(time.time())
    start_time = end_time - (60 * 60 * 6)  # last 6 hours of data
    prometheus_url = "http://203.175.10.196:9090/api/v1/query_range"

    # CPU usage query
    query_cpu = '100 * (1 - avg by(job)(irate(node_cpu_seconds_total{job=~"SIEM", mode="idle"}[1m])))'
    params_cpu = {
        "query": query_cpu,
        "start": start_time,
        "end": end_time,
        "step": step,
    }

    response = requests.get(prometheus_url, params=params_cpu)
    data = response.json()

    try:
        result = data["data"]["result"][0]["values"]
    except (KeyError, IndexError):
        raise ValueError("Prometheus CPU query did not return any data.")

    # Example: extracting the last 20 points (adapt to however your model was trained)
    window_values = [float(point[1]) for point in result[-20:]]

    # Scale the values
    window_values = np.array(window_values).reshape(-1, 1)
    window_values_scaled = scaler_cpu.transform(window_values)

    # Reshape for the model (batch, timesteps, features)
    seed = window_values_scaled.reshape(1, -1, 1)
    return seed


def get_mem_data(step: int):
    """
    Retrieves recent Prometheus Memory usage data for 'SIEM'.
    Returns the scaled seed array matching the Memory model input shape.
    """
    end_time = int(time.time())
    start_time = end_time - (60 * 60 * 2)  # last 2 hours of data
    prometheus_url = "http://203.175.10.196:9090/api/v1/query_range"

    # Memory usage query
    query_mem = (
        '(node_memory_MemTotal_bytes{job="SIEM"} - node_memory_MemAvailable_bytes{job="SIEM"}) '
        '* 100 / node_memory_MemTotal_bytes{job="SIEM"}'
    )
    params_mem = {
        "query": query_mem,
        "start": start_time,
        "end": end_time,
        "step": step,
    }

    response = requests.get(prometheus_url, params=params_mem)
    data = response.json()

    try:
        result = data["data"]["result"][0]["values"]
    except (KeyError, IndexError):
        raise ValueError("Prometheus Memory query did not return any data.")

    # Example: extracting the last 4 points for a 4-step window model
    window_values = [float(point[1]) for point in result[-4:]]

    # Scale the values
    window_values = np.array(window_values).reshape(-1, 1)
    window_values_scaled = scaler_mem.transform(window_values)

    # Reshape for the model (batch, timesteps, features)
    seed = window_values_scaled.reshape(1, -1, 1)
    return seed


def direct_forecast(model, seed, scaler):
    """
    Predicts one time step ahead using the pre-trained model and returns
    the prediction in the original scale (inverse-transformed).
    """
    pred_scaled = model.predict(seed)  # shape: (1, 1)
    pred_original = scaler.inverse_transform(pred_scaled)
    return float(pred_original[0, 0])


# ---------------------------
# 4) Endpoints
# ---------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the LSTM Forecasting API!"}


@app.get("/test_forecast_cpu")
def test_forecast_cpu(step: int = 1800):
    """
    Endpoint for CPU forecasting.
    Pass a step interval (seconds) to control the Prometheus query resolution.
    Default is 1800s (30 minutes).
    """
    seed = get_cpu_data(step)
    forecast_value = direct_forecast(model_bilstm_cpu, seed, scaler_cpu)
    return {
        "message": "CPU Forecast Successful",
        "Prediction (next step)": forecast_value,
    }


@app.get("/test_forecast_mem")
def test_forecast_mem(step: int = 1800):
    """
    Endpoint for Memory forecasting.
    Pass a step interval (seconds) to control the Prometheus query resolution.
    Default is 1800s (30 minutes).
    """
    seed = get_mem_data(step)
    forecast_value = direct_forecast(model_bilstm_mem, seed, scaler_mem)
    return {
        "message": "Memory Forecast Successful",
        "Prediction (next step)": forecast_value,
    }


# ---------------------------
# 5) Main
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
