"""
Flight Delay Prediction API with SHAP Explainability (Explainable AI Version)
"""

from flask import Flask, render_template, request, jsonify
import joblib
import requests
import json
from datetime import datetime
import logging
import os
import time
from functools import lru_cache
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Lazy load SHAP to avoid import issues
shap = None

# ============================================================================
# SETUP
# ============================================================================

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# ============================================================================
# LOAD MODEL & DATA ON STARTUP
# ============================================================================

def load_model():
    try:
        model_data = {
            "model": joblib.load(os.path.join(BASE_DIR, "../models/airline_delay_lgbm.pkl")),
            "feature_names": joblib.load(os.path.join(BASE_DIR, "../models/feature_names.pkl")),
            "optimal_threshold": joblib.load(os.path.join(BASE_DIR, "../models/optimal_threshold.pkl")),
        }
        logger.info("✅ Model loaded successfully")
        return model_data
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return None

def load_background_data():
    """Load background data for SHAP explainer"""
    try:
        # Load training data for SHAP background (use sample for performance)
        X_train_file = os.path.join(BASE_DIR, "../data/processed/X_train_weather.csv")
        if os.path.exists(X_train_file):
            X_train = pd.read_csv(X_train_file)
            # Use random sample for faster SHAP (100 samples is good)
            background = X_train.sample(n=min(100, len(X_train)), random_state=42)
            logger.info(f"✅ Background data loaded: {len(background)} samples")
            return background
        else:
            logger.warning("⚠️ Background data file not found")
            return None
    except Exception as e:
        logger.error(f"❌ Background data loading failed: {e}")
        return None

MODEL_DATA = load_model()
BACKGROUND_DATA = load_background_data()

MODEL_DATA = load_model()
BACKGROUND_DATA = load_background_data()

# Create SHAP explainer (lazy load - only when first prediction is made)
SHAP_EXPLAINER = None

def get_shap_explainer():
    """Lazy load SHAP explainer on first use"""
    global shap, SHAP_EXPLAINER
    
    if SHAP_EXPLAINER is not None:
        return SHAP_EXPLAINER
    
    if not MODEL_DATA or BACKGROUND_DATA is None:
        logger.warning("⚠️ SHAP unavailable: Model or background data not loaded")
        return None
    
    try:
        if shap is None:
            import shap as shap_module
            shap = shap_module
            logger.info("✅ SHAP library imported")
        
        SHAP_EXPLAINER = shap.TreeExplainer(MODEL_DATA["model"])
        logger.info("✅ SHAP explainer created successfully")
        return SHAP_EXPLAINER
    except Exception as e:
        logger.warning(f"⚠️ SHAP explainer creation failed: {e}")
        return None

# ============================================================================
# DATA
# ============================================================================

AIRPORTS = {
    "JFK": {"lat": 40.6413, "lon": -73.7781},
    "LAX": {"lat": 33.9425, "lon": -118.4081},
    "ORD": {"lat": 41.9742, "lon": -87.9073},
    "ATL": {"lat": 33.6407, "lon": -84.4277},
    "DFW": {"lat": 32.8975, "lon": -97.0382},
    "DEN": {"lat": 39.8561, "lon": -104.6737},
    "SFO": {"lat": 37.6213, "lon": -122.3790},
    "MIA": {"lat": 25.7959, "lon": -80.2870},
    "BOS": {"lat": 42.3656, "lon": -71.0096},
}

AIRLINE_MAP = {
    "AA": 1, "DL": 2, "UA": 3, "SW": 4,
    "JB": 5, "NK": 6, "F9": 7
}

ORIGIN_MAP = DEST_MAP = {
    "JFK": 1, "LAX": 2, "ORD": 3, "ATL": 4,
    "DFW": 5, "DEN": 6, "SFO": 7, "MIA": 8, "BOS": 9
}

# ============================================================================
# WEATHER (CACHED)
# ============================================================================

@lru_cache(maxsize=100)
def get_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,precipitation,wind_speed_10m,pressure_msl"
        response = requests.get(url, timeout=3)
        data = response.json()["current"]

        return {
            "temp": data.get("temperature_2m", 15),
            "precip": data.get("precipitation", 0),
            "wind_speed": data.get("wind_speed_10m", 10),
            "pressure": data.get("pressure_msl", 1013),
        }
    except Exception as e:
        logger.warning(f"Weather API failed: {e}")
        return {"temp": 15, "precip": 0, "wind_speed": 10, "pressure": 1013}

# ============================================================================
# DISTANCE (HAVERSINE)
# ============================================================================

def calculate_distance(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, asin, sqrt

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3959

    return c * r

# ============================================================================
# PREPROCESS
# ============================================================================

def preprocess_input(data):
    origin = data["origin"]
    dest = data["dest"]
    airline = data["airline"]
    dep_datetime = datetime.strptime(
        f"{data['dep_date']} {data['dep_time']}", "%Y-%m-%d %H:%M"
    )

    origin_w = get_weather(*AIRPORTS[origin].values())
    dest_w = get_weather(*AIRPORTS[dest].values())

    features = {
        "ORIGIN": ORIGIN_MAP[origin],
        "DEST": DEST_MAP[dest],
        "OP_UNIQUE_CARRIER": AIRLINE_MAP[airline],
        "YEAR": dep_datetime.year,
        "MONTH": dep_datetime.month,
        "DAY_OF_WEEK": dep_datetime.weekday(),
        "CRS_DEP_HOUR": dep_datetime.hour,
        "DISTANCE": data["distance"],
        "origin_tavg": origin_w["temp"],
        "origin_prcp": origin_w["precip"],
        "origin_wspd": origin_w["wind_speed"],
        "origin_pres": origin_w["pressure"],
        "dest_tavg": dest_w["temp"],
        "dest_prcp": dest_w["precip"],
        "dest_wspd": dest_w["wind_speed"],
        "dest_pres": dest_w["pressure"],
    }

    # Derived features
    hour = features["CRS_DEP_HOUR"]

    features["is_peak_hour"] = int(hour in [7,8,9,16,17,18,19,20])
    features["is_late_flight"] = int(hour >= 21)

    features["temp_diff"] = abs(features["origin_tavg"] - features["dest_tavg"])

    return features

# ============================================================================
# PREDICTION WITH SHAP
# ============================================================================

def make_prediction(features):
    try:
        start = time.time()

        model = MODEL_DATA["model"]
        feature_names = MODEL_DATA["feature_names"]
        threshold = MODEL_DATA["optimal_threshold"]

        df = pd.DataFrame([features])

        for col in feature_names:
            if col not in df:
                df[col] = 0

        df = df[feature_names]

        prob = model.predict_proba(df)[0][1]
        pred = "DELAYED" if prob >= threshold else "ON-TIME"

        # ===== SHAP EXPLAINABILITY =====
        explanation = None
        shap_explainer = get_shap_explainer()
        
        if shap_explainer is not None:
            try:
                # Calculate SHAP values
                shap_values = shap_explainer.shap_values(df)
                
                # Get base value (expected model output)
                base_value = shap_explainer.expected_value
                if isinstance(base_value, list):
                    base_value = base_value[1]  # For binary classification, take delay class
                
                # Get feature contributions (SHAP values for delayed class)
                shap_vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
                
                # Create feature importance list
                feature_importance = []
                for fname, shap_val in zip(feature_names, shap_vals):
                    feature_importance.append({
                        "feature": fname,
                        "value": float(df[fname].values[0]),
                        "shap_value": float(shap_val),
                        "contribution": "INCREASES DELAY" if shap_val > 0 else "DECREASES DELAY",
                        "impact": abs(float(shap_val))
                    })
                
                # Sort by absolute impact
                feature_importance.sort(key=lambda x: x["impact"], reverse=True)
                
                # Keep top 5 features
                top_features = feature_importance[:5]
                
                explanation = {
                    "base_value": float(base_value),
                    "top_features": top_features,
                    "model_confidence": float(prob)
                }
                
            except Exception as e:
                logger.warning(f"⚠️ SHAP calculation failed: {e}")
                explanation = None

        prediction_time = time.time() - start
        logger.info(f"Prediction took {prediction_time:.2f}s")

        result = {
            "prediction": pred,
            "probability": float(prob),
            "confidence": round(max(prob, 1 - prob) * 100, 2),
            "explanation": explanation
        }

        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

# ============================================================================
# ROUTES
# ============================================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validation
        required = ["origin", "dest", "airline", "dep_date", "dep_time"]

        if not data:
            return jsonify({"error": "No input data"}), 400

        for field in required:
            if field not in data or not data[field]:
                return jsonify({"error": f"{field} required"}), 400

        if data["origin"] not in AIRPORTS or data["dest"] not in AIRPORTS:
            return jsonify({"error": "Invalid airport"}), 400

        # Distance
        o = AIRPORTS[data["origin"]]
        d = AIRPORTS[data["dest"]]

        distance = calculate_distance(o["lat"], o["lon"], d["lat"], d["lon"])
        data["distance"] = distance

        features = preprocess_input(data)
        result = make_prediction(features)

        return jsonify({
            "success": True,
            "distance": round(distance, 2),
            "result": result
        })

    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3001)  # Different port (3001 instead of 3000)
