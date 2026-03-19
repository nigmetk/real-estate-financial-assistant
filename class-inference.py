import joblib
import json
import numpy as np
import os

# Load model, scaler, and feature order
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "logistic_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    features_path = os.path.join(model_dir, "features.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_order = joblib.load(features_path)

    return {"model": model, "scaler": scaler, "feature_order": feature_order}

# Parse input
def input_fn(request_body, content_type):
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")

    if content_type == "application/json":
        data = json.loads(request_body)
        return data
    else:
        raise ValueError("Unsupported content type")

# Prediction
def predict_fn(input_data, model_dict):
    model = model_dict["model"]
    scaler = model_dict["scaler"]
    feature_order = model_dict["feature_order"]

    # Rebuild feature vector in correct order
    ordered_features = []
    for f in feature_order:
        ordered_features.append(input_data.get(f, 0))

    X = np.array(ordered_features).reshape(1, -1)

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    pred = int(model.predict(X_scaled)[0])
    prob = float(model.predict_proba(X_scaled)[0][1])

    return {"prediction": pred, "probability": prob}

# Output
def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
