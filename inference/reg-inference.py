import joblib
import json
import numpy as np
import os


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "random_forest_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, content_type):
    content_type = content_type or "application/json"

    if content_type.startswith("application/json"):
        # SageMaker bazen bytes gönderir → decode etmek şart
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")

        data = json.loads(request_body)
        features = data["features"]
        return np.array(features).reshape(1, -1)

    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    return prediction.tolist()


def output_fn(prediction, accept):
    accept = accept or "application/json"

    if accept.startswith("application/json"):
        return json.dumps({"prediction": prediction}), "application/json"
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
