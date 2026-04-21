# ✈️ Flight Delay Predictor - Flask Web App

A production-ready Flask web application for predicting flight delays using machine learning and real-time weather data.

## 📋 Features

- **🎯 ML-Powered Predictions**: Uses Optuna-tuned LightGBM model (80% accuracy)
- **🌦️ Real-Time Weather**: Auto-fetches current weather from Open-Meteo API (free, no key needed)
- **✨ Beautiful UI**: Responsive, modern interface built with HTML/CSS/JS
- **⚡ RESTful API**: Easy integration with other applications
- **📊 Model Transparency**: Shows confidence, probabilities, and model stats

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Model Files

Make sure these files exist in `../models/`:
- `airline_delay_lgbm.pkl` — trained model
- `feature_names.pkl` — feature list
- `optimal_threshold.pkl` — decision threshold
- `ordinal_encoder.pkl` — categorical encoder
- `numeric_imputer.pkl` — numeric imputer

### 3. Run the App

```bash
python app.py
```

The app will start at **http://localhost:5000**

## 📖 How to Use

1. **Select Flight Details**:
   - Origin & destination airports
   - Airline
   - Departure date & time
   - Distance (auto-filled for major routes)

2. **Click "Predict Delay"**:
   - App fetches real-time weather
   - Model analyzes all factors
   - Shows prediction in ~2 seconds

3. **View Results**:
   - Prediction: "ON-TIME" or "DELAYED"
   - Confidence percentage
   - Model stats

## 🔌 API Endpoints

### POST `/api/predict`

**Request:**
```json
{
  "origin": "JFK",
  "dest": "LAX",
  "airline": "AA",
  "dep_date": "2024-04-01",
  "dep_time": "14:30",
  "distance": 2451
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "prediction": "DELAYED",
    "probability": 0.65,
    "confidence": 65.0,
    "threshold": 0.577
  }
}
```

### GET `/api/model-info`

Returns model metadata (accuracy, F1, AUC, etc.)

## 🧠 Model Details

| Metric | Value |
|--------|-------|
| **Model** | LightGBM (Optuna-tuned) |
| **Features** | 49 (weather, historical rates, temporal, spatial) |
| **Test Accuracy** | 80.01% |
| **Test F1 Score** | 0.4892 |
| **ROC-AUC** | 0.7741 |
| **Decision Threshold** | 0.577 |

## 📦 Deployment

### Local Development
```bash
python app.py  # http://localhost:5000
```

### Production (Heroku)

1. Create `Procfile`:
```
web: gunicorn app:app
```

2. Add `gunicorn` to `requirements.txt`

3. Deploy:
```bash
heroku login
heroku create your-app-name
git push heroku main
```

### Production (AWS/Azure/DigitalOcean)

Use Docker:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🌦️ Weather API

Uses **Open-Meteo API** (free, no authentication):
- Current temperature
- Precipitation
- Wind speed
- Atmospheric pressure

**Example call:**
```
https://api.open-meteo.com/v1/forecast?latitude=40.6413&longitude=-73.7781&current=temperature_2m,precipitation,wind_speed_10m,pressure_msl
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: models/airline_delay_lgbm.pkl` | Ensure model files exist in `../models/` |
| Weather API timeout | App uses default values; predictions still work |
| CORS errors (if calling from another domain) | Add `flask-cors`: `pip install flask-cors` then `from flask_cors import CORS` and `CORS(app)` |

## 📝 File Structure

```
flight_delay_app/
├── app.py                    # Flask backend
├── templates/
│   └── index.html           # Frontend UI
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔐 Security Notes

- For production: Set `debug=False`
- Add authentication if needed
- Implement rate limiting to prevent API abuse
- Store sensitive config in environment variables

## 📄 License

This project uses the trained model from `model_comparison_v3.ipynb`

## 👨‍💻 Author

Built for flight delay prediction project (Sem 8, Mini-Project)

---

**Questions?** Check model_comparison_v3.ipynb for model details, or see explainability.ipynb for feature importance.
