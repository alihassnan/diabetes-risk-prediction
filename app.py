
from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
import json

app = Flask(__name__, static_folder="templates")  # frontend folder

# Load scaler & model
scaler = joblib.load("scaler.joblib")
xgb_model = joblib.load("diabetes_xgb_model.pkl")

# Optional: feature names from JSON (agar separate file hai)
with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

# Serve frontend
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

# Predict API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # Default 0 if missing
    features = [float(data.get(f, 0)) for f in feature_names]
    scaled_features = scaler.transform([features])
    prediction = int(xgb_model.predict(scaled_features)[0])
    return jsonify({"prediction": prediction})

# Static files route (optional)
@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

