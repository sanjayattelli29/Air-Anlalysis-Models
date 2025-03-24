from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models and preprocessing tools
models = {
    "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
    "KNN": joblib.load("knn_model.pkl"),
    "SVM": joblib.load("svm_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl")
}
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define feature names as per the dataset
FEATURE_NAMES = [
    "pm25", "pm10", "no", "no2", "nox", "nh3", "so2", "co",
    "o3", "benzene", "humidity", "wind_speed", "wind_direction",
    "solar_radiation", "rainfall", "air_temperature"
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Expecting JSON input

        # Validate input
        if "features" not in data or not isinstance(data["features"], list) or len(data["features"]) != 16:
            return jsonify({"error": "Invalid input. Expected 16 feature values in a list."}), 400

        # Convert input to DataFrame
        df_input = pd.DataFrame([data["features"]], columns=FEATURE_NAMES)

        # Scale input
        X_scaled = scaler.transform(df_input)

        # Get predictions from all models
        predictions = {}
        for model_name, model in models.items():
            y_pred = model.predict(X_scaled)
            y_pred_label = label_encoder.inverse_transform(y_pred)[0]
            predictions[model_name] = y_pred_label

        # Final Recommendation (most frequent predicted category)
        final_recommendation = max(predictions.values(), key=predictions.values().count)

        return jsonify({
            "predictions": predictions,
            "final_recommendation": final_recommendation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
