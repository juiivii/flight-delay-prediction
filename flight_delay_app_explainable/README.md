# 🤖 Flight Delay Predictor - Explainable AI Version

## Overview

This is the **EXPLAINABLE AI VERSION** of the Flight Delay Prediction system. It includes SHAP (SHapley Additive exPlanations) feature importance to show **WHY** each prediction is made.

**Location:** `/flight_delay_app_explainable/`
**Port:** 3001 (runs separately from basic version)

---

## Key Differences from Basic Version

| Feature | Basic App | Explainable App |
|---------|-----------|-----------------|
| Prediction | ✅ | ✅ |
| Confidence Score | ✅ | ✅ |
| Distance | ✅ | ✅ |
| **SHAP Explanations** | ❌ | ✅ **NEW!** |
| **Top 5 Contributing Features** | ❌ | ✅ **NEW!** |
| **Feature Impact Visualization** | ❌ | ✅ **NEW!** |
| Response Time | ~50ms | ~300-500ms |
| Port | 3000 | 3001 |

---

## What's New: SHAP Explanations

### Backend Changes (app.py)

**Added:**
- SHAP TreeExplainer initialization
- Background data loading (for SHAP reference)
- SHAP value calculation for each prediction
- Feature importance extraction
- JSON response with explanations

**Code Additions:**
```python
# Load background data for SHAP
BACKGROUND_DATA = load_background_data()
SHAP_EXPLAINER = shap.TreeExplainer(MODEL_DATA["model"])

# In make_prediction():
shap_values = SHAP_EXPLAINER.shap_values(df)
# Extract top 5 features with SHAP values
# Return as JSON
```

### Frontend Changes (index.html)

**Added:**
- Explanation section in result display
- Top 5 features with contributions
- Impact visualization (bars)
- "Increases Delay" vs "Decreases Delay" indicators
- Better styling and layout

---

## How It Works

### 1. User Makes Prediction
```
User inputs: JFK → LAX, American Airlines, April 1, 2:00 PM
```

### 2. Model Predicts
```
Model: "68% chance of delay"
```

### 3. SHAP Explains
```
🧠 Why This Prediction?

1. carrier_route_delay_rate: 0.32
   📈 INCREASES DELAY (Impact: 0.18)
   
2. dest_wspd: 18.5
   📈 INCREASES DELAY (Impact: 0.12)
   
3. is_peak_hour: 1.0
   📈 INCREASES DELAY (Impact: 0.08)
   
4. distance: 247.3
   📉 DECREASES DELAY (Impact: -0.08)
   
5. origin_wspd: 8.2
   📉 DECREASES DELAY (Impact: -0.02)
```

---

## Running the Explainable App

### 1. Install Dependencies
```bash
cd flight_delay_app_explainable
pip install -r requirements.txt
```

### 2. Install SHAP
```bash
pip install shap
```

### 3. Run the App
```bash
python app.py
```

The app will start on:
- **http://127.0.0.1:3001**
- **http://localhost:3001**

---

## Understanding the Output

### Example Output

```json
{
    "success": true,
    "distance": 247.3,
    "result": {
        "prediction": "DELAYED",
        "probability": 0.68,
        "confidence": 68.0,
        "explanation": {
            "base_value": 0.18,
            "top_features": [
                {
                    "feature": "carrier_route_delay_rate",
                    "value": 0.32,
                    "shap_value": 0.18,
                    "contribution": "INCREASES DELAY",
                    "impact": 0.18
                },
                {
                    "feature": "dest_wspd",
                    "value": 18.5,
                    "shap_value": 0.12,
                    "contribution": "INCREASES DELAY",
                    "impact": 0.12
                },
                ...
            ],
            "model_confidence": 0.68
        }
    }
}
```

---

## Performance Notes

### Speed Comparison

| Version | Prediction Time | Explanation Time | Total |
|---------|-----------------|------------------|-------|
| Basic | ~50ms | N/A | ~100ms |
| Explainable | ~50ms | 200-500ms | **300-600ms** |

**Note:** SHAP calculations add 200-500ms delay. Still fast enough for web app!

### Memory Usage

| Component | Size |
|-----------|------|
| Model | ~50MB |
| Background data (100 samples) | ~5MB |
| SHAP Explainer | ~10MB |
| Total overhead | ~15MB |

---

## File Structure

```
flight_delay_app_explainable/
├── app.py                    # Flask app with SHAP
├── templates/
│   └── index.html           # Frontend with explanation display
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## Key Features Explained

### 1. SHAP Values
- **What:** Shapley values from game theory
- **Why:** Fair attribution of prediction to features
- **How:** Shows each feature's contribution to the final prediction

### 2. Top 5 Features
- **What:** Most influential features for this specific flight
- **Why:** Users don't need all 39 features, just the important ones
- **How:** Ranked by absolute SHAP value

### 3. Impact Visualization
- **What:** Colored bars showing relative importance
- **Why:** Visual representation is easier to understand
- **How:** Bar width proportional to SHAP value magnitude

### 4. "Increases/Decreases Delay"
- **What:** Direction of feature's influence
- **Why:** Tells user if feature pushes toward or away from delay
- **How:** Positive SHAP = increases delay, Negative = decreases

---

## Example Predictions

### Flight 1: Likely On-Time
```
✅ ON-TIME (75% confidence)

Why?
📉 Long distance (-8%)
📉 Good weather (-5%)
📉 Off-peak hour (-3%)
```

### Flight 2: Likely Delayed
```
⚠️ DELAYED (68% confidence)

Why?
📈 Bad historical record (+18%)
📈 Wind at destination (+12%)
📈 Peak hour (+8%)
```

---

## Troubleshooting

### SHAP Installation Issues
```bash
# If shap fails to install:
pip install shap --no-binary shap
```

### Background Data Not Found
```
⚠️ Warning: Background data file not found
- SHAP explainer will be disabled
- Predictions will still work (without explanations)
- Make sure X_train_weather.csv exists
```

### Slow Predictions
```
- Normal: SHAP calculations add 200-500ms
- If > 1 second, reduce background sample size in code
- Change: background = X_train.sample(n=50) # was 100
```

---

## Comparison: Basic vs Explainable

### Basic App (Port 3000)
```
Input → Prediction → Output
         
Shows: ON-TIME, 68%, Distance
Uses: ~100ms, ~250MB memory
```

### Explainable App (Port 3001)
```
Input → Prediction → SHAP Calculation → Output
                    ↓
         Explanation with top 5 features
         
Shows: ON-TIME, 68%, Distance, SHAP Explanation
Uses: ~600ms, ~265MB memory
```

---

## Running Both Versions Simultaneously

You can run both apps at the same time:

```bash
# Terminal 1: Basic version
cd flight_delay_app
python app.py  # Runs on port 3000

# Terminal 2: Explainable version
cd flight_delay_app_explainable
python app.py  # Runs on port 3001
```

Then access:
- Basic: http://localhost:3000
- Explainable: http://localhost:3001

---

## Next Steps

1. ✅ Test the app and verify predictions
2. ✅ Check SHAP explanations are clear
3. ✅ Optimize for faster SHAP if needed
4. ✅ Deploy to production
5. ✅ Get user feedback on explanations

---

## Questions?

- **Why SHAP?** → Theoretically sound + easy to understand
- **Why top 5?** → Balance between clarity and completeness
- **Why so slow?** → SHAP calculations are computationally expensive (normal)
- **Can we speed it up?** → Yes, use smaller background data or simplify model

---

**Version:** 1.0 (Explainable AI)
**Date:** April 1, 2026
**Status:** ✅ Ready for Testing
