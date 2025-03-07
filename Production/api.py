import time

import joblib
import requests
import tensorflow as tf
import uvicorn
from fastapi import FastAPI
from keras.saving import register_keras_serializable


@register_keras_serializable(package="CustomLoss", name="loss_fn")
def loss_fn(y_true, y_pred):
    alpha = 4.5
    error = y_pred - y_true
    sq_err = tf.square(error)
    mask_under = tf.less(error, 0.0)
    loss = tf.where(mask_under, alpha * sq_err, sq_err)
    return tf.reduce_mean(loss)


# -------------------------
# 1) Load All Models & Scalers
# -------------------------

base_path = "/home/fauzan/forecasting-infrastructure-git/Production"

# APM
model_bilstm_cpu_APM = tf.keras.models.load_model(
    f"{base_path}/APM/CPU/my_model-bilstm-window_12.keras"
)
scaler_cpu_APM = joblib.load(f"{base_path}/APM/CPU/scaler-cpu.pkl")

model_bilstm_mem_APM = tf.keras.models.load_model(
    f"{base_path}/APM/Memory/my_model-gru-window_12.keras"
)
scaler_mem_APM = joblib.load(f"{base_path}/APM/Memory/scaler-memo.pkl")

model_bilstm_disk_APM = tf.keras.models.load_model(
    f"{base_path}/APM/Disk/my_model-gru-window_12.keras"
)
scaler_disk_APM = joblib.load(f"{base_path}/APM/Disk/scaler.pkl")

# SIEM
model_bilstm_cpu_SIEM = tf.keras.models.load_model(
    f"{base_path}/SIEM/CPU/my_model-bilstm-window_12.keras"
)
scaler_cpu_SIEM = joblib.load(f"{base_path}/SIEM/CPU/scaler-cpu.pkl")

model_bilstm_mem_SIEM = tf.keras.models.load_model(
    f"{base_path}/SIEM/Memory/my_model-bilstm-window_4.keras"
)
scaler_mem_SIEM = joblib.load(f"{base_path}/SIEM/Memory/scaler-memo.pkl")

model_bilstm_disk_SIEM = tf.keras.models.load_model(
    f"{base_path}/SIEM/Disk/my_model-bilstm-window_12.keras"
)
scaler_disk_SIEM = joblib.load(f"{base_path}/SIEM/Disk/scaler.pkl")

# KT
model_bilstm_cpu_KT = tf.keras.models.load_model(
    f"{base_path}/KejarTugas/CPU/my_model-gru-window_12.keras"
)
scaler_cpu_KT = joblib.load(f"{base_path}/KejarTugas/CPU/scaler.pkl")

model_bilstm_mem_KT = tf.keras.models.load_model(
    f"{base_path}/KejarTugas/Memory/my_model-gru-window_12.keras"
)
scaler_mem_KT = joblib.load(f"{base_path}/KejarTugas/Memory/scaler.pkl")

model_bilstm_disk_KT = tf.keras.models.load_model(
    f"{base_path}/KejarTugas/Disk/my_model-gru-window_12.keras"
)
scaler_disk_KT = joblib.load(f"{base_path}/KejarTugas/Disk/scaler.pkl")


# -------------------------
# 2) Configuration Dictionaries
# -------------------------
# Each server/metric pair references:
#   - a Keras model
#   - a scaler
#   - a "look_back_window"
#   - a query template (with placeholders for the server)
server_configs = {
    "APM": {
        "cpu": {
            "model": model_bilstm_cpu_APM,
            "scaler": scaler_cpu_APM,
            "window": 12,
            "query_template": '100 * (1 - avg by(job)(irate(node_cpu_seconds_total{{job="{server}", mode="idle"}}[1m])))',
        },
        "mem": {
            "model": model_bilstm_mem_APM,
            "scaler": scaler_mem_APM,
            "window": 12,
            "query_template": '(node_memory_MemTotal_bytes{{job="{server}"}} - node_memory_MemAvailable_bytes{{job="{server}"}}) * 100 / node_memory_MemTotal_bytes{{job="{server}"}}',
        },
        "disk": {
            "model": model_bilstm_disk_APM,
            "scaler": scaler_disk_APM,
            "window": 12,
            "query_template": '((node_filesystem_size_bytes{{job="{server}", mountpoint="/"}} - node_filesystem_free_bytes{{job="{server}", mountpoint="/"}}) / node_filesystem_size_bytes{{job="{server}", mountpoint="/"}}) * 100',
        },
    },
    "SIEM": {
        "cpu": {
            "model": model_bilstm_cpu_SIEM,
            "scaler": scaler_cpu_SIEM,
            "window": 12,
            "query_template": '100 * (1 - avg by(job)(irate(node_cpu_seconds_total{{job="{server}", mode="idle"}}[1m])))',
        },
        "mem": {
            "model": model_bilstm_mem_SIEM,
            "scaler": scaler_mem_SIEM,
            "window": 4,
            "query_template": '(node_memory_MemTotal_bytes{{job="{server}"}} - node_memory_MemAvailable_bytes{{job="{server}"}}) * 100 / node_memory_MemTotal_bytes{{job="{server}"}}',
        },
        "disk": {
            "model": model_bilstm_disk_SIEM,
            "scaler": scaler_disk_SIEM,
            "window": 12,
            "query_template": '((node_filesystem_size_bytes{{job="{server}", mountpoint="/"}} - node_filesystem_free_bytes{{job="{server}", mountpoint="/"}}) / node_filesystem_size_bytes{{job="{server}", mountpoint="/"}}) * 100',
        },
    },
    "ktprod": {
        "cpu": {
            "model": model_bilstm_cpu_KT,
            "scaler": scaler_cpu_KT,
            "window": 12,
            "query_template": '100 * (1 - avg by(job)(irate(node_cpu_seconds_total{{job="{server}", mode="idle"}}[1m])))',
        },
        "mem": {
            "model": model_bilstm_mem_KT,
            "scaler": scaler_mem_KT,
            "window": 12,
            "query_template": '(node_memory_MemTotal_bytes{{job="{server}"}} - node_memory_MemAvailable_bytes{{job="{server}"}}) * 100 / node_memory_MemTotal_bytes{{job="{server}"}}',
        },
        "disk": {
            "model": model_bilstm_disk_KT,
            "scaler": scaler_disk_KT,
            "window": 12,
            "query_template": '((node_filesystem_size_bytes{{job="{server}", mountpoint="/"}} - node_filesystem_free_bytes{{job="{server}", mountpoint="/"}}) / node_filesystem_size_bytes{{job="{server}", mountpoint="/"}}) * 100',
        },
    },
}


# -------------------------
# 3) Helper Functions
# -------------------------
def fetch_prometheus_data(query: str, minutes: int = 30):
    """
    Fetches data from Prometheus for the given query, returning the last 'minutes' data points.
    """
    end_time = int(time.time())
    start_time = end_time - (60 * 30 * 12)
    step = 60 * 30  # one data point per minute

    prometheus_url = "http://203.175.10.196:9090/api/v1/query_range"
    params = {"query": query, "start": start_time, "end": end_time, "step": step}
    response = requests.get(prometheus_url, params=params)
    data = response.json()
    # Extract the relevant values from result
    try:
        result = data["data"]["result"][0]["values"]
    except (KeyError, IndexError):
        raise ValueError("Prometheus query did not return any data.")

    # Convert the last points to float
    values = [float(point[1]) for point in result]
    return values


def direct_forecast(model, scaler, series: list, look_back: int):
    """
    Given a univariate time series (list of floats), create a single look-back window,
    scale it, predict one step, and inverse-transform.
    """
    import numpy as np

    # 1) Make sure we have enough data points
    if len(series) < look_back:
        raise ValueError(
            f"Not enough data points ({len(series)}) for look_back={look_back}."
        )

    # 2) Extract the last 'look_back' points for the seed
    window_values = series[-look_back:]  # shape: (look_back, )

    # 3) Reshape to (look_back, 1) for scaler
    arr_2d = np.array(window_values).reshape(-1, 1)  # shape: (look_back, 1)
    scaled_2d = scaler.transform(arr_2d)

    # 4) Reshape to (1, look_back, 1) for the model
    seed = scaled_2d.reshape(1, look_back, 1)

    # 5) Predict one step
    pred_scaled = model.predict(seed)  # shape: (1, 1)
    # 6) Inverse transform
    pred_original = scaler.inverse_transform(pred_scaled)

    return float(pred_original[0, 0])


# -------------------------
# 4) FastAPI App
# -------------------------
app = FastAPI(title="Deep Learning Forecast API")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Forecast API!"}


@app.get("/forecast")
def forecast(server: str, metric: str):
    """
    Example usage:
      GET /forecast?server=APM&metric=cpu
      GET /forecast?server=SIEM&metric=mem
    """
    if server not in server_configs:
        return {
            "error": f"Unknown server '{server}' (must be one of {list(server_configs.keys())})."
        }
    if metric not in server_configs[server]:
        return {
            "error": f"Unknown metric '{metric}' for server '{server}' (must be one of {list(server_configs[server].keys())})."
        }

    config = server_configs[server][metric]
    model = config["model"]
    scaler = config["scaler"]
    window = config["window"]

    query_str = config["query_template"].format(server=server)

    try:
        series = fetch_prometheus_data(query_str, minutes=120)
    except ValueError as e:
        return {"error": str(e)}

    # Forecast one step
    try:
        forecast_val = direct_forecast(model, scaler, series, look_back=window)
    except ValueError as e:
        return {"error": str(e)}

    return {"server": server, "metric": metric, "forecast": 100 - forecast_val}


@app.get("/all_forecast")
def all_forecast(metric: str):
    """
    Example usage:
      GET /all_forecast?metric=cpu
      GET /all_forecast?metric=mem
      GET /all_forecast?metric=disk

    This endpoint fetches predictions from APM, SIEM, and ktprod for the given metric
    and returns both the individual forecast values and an overall average.
    """
    servers = ["APM", "SIEM", "ktprod"]

    # Validate that at least one server has this metric
    # (or you can skip this if you're sure the metric exists for each server)
    valid_metric_anywhere = any(metric in server_configs[s] for s in servers)
    if not valid_metric_anywhere:
        return {"error": f"Unknown metric '{metric}'."}

    results = {}
    sum_forecasts = 0.0
    valid_count = 0

    for server in servers:
        if metric not in server_configs[server]:
            # If that server doesn't support the metric, skip or mark as "Unsupported"
            results[server] = "Unsupported"
            continue

        config = server_configs[server][metric]
        model = config["model"]
        scaler = config["scaler"]
        window = config["window"]

        # Build the query
        query_str = config["query_template"].format(server=server)

        try:
            # Fetch data from Prometheus
            series = fetch_prometheus_data(query_str, minutes=120)

            # Forecast one step
            forecast_val = direct_forecast(model, scaler, series, look_back=window)
            # results[server] = forecast_val

            sum_forecasts += forecast_val
            valid_count += 1

        except ValueError as e:
            results[server] = f"Error: {str(e)}"
        except Exception as e:
            results[server] = f"Unexpected error: {str(e)}"

    # Compute average of all valid forecasts
    if valid_count > 0:
        average_val = sum_forecasts / valid_count
    else:
        average_val = "No valid forecasts"

    results["average"] = 100 - average_val
    return results

    """
    For Disk, gather forecasts for servers: APM, SIEM, ktprod.
    Then compute an average of the three forecasts.
    Return a JSON with individual forecasts and the overall average.
    """
    servers = ["APM", "SIEM", "ktprod"]
    metric = "disk"

    results = {}
    sum_forecasts = 0.0
    valid_count = 0

    for server in servers:
        if server not in server_configs or metric not in server_configs[server]:
            results[server] = "Unsupported"
            continue

        config = server_configs[server][metric]
        model = config["model"]
        scaler = config["scaler"]
        window = config["window"]
        query_str = config["query_template"].format(server=server)

        try:
            series = fetch_prometheus_data(query_str, minutes=120)
            forecast_val = direct_forecast(model, scaler, series, look_back=window)
            # results[server] = forecast_val
            sum_forecasts += forecast_val
            valid_count += 1
        except ValueError as e:
            results[server] = f"Error: {str(e)}"
        except Exception as e:
            results[server] = f"Unexpected error: {str(e)}"

    if valid_count > 0:
        average_val = sum_forecasts / valid_count
    else:
        average_val = "No valid forecasts"

    results["average"] = average_val
    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
