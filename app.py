from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained models
models = {
    "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
    "KNN": joblib.load("knn_model.pkl"),
    "SVM": joblib.load("svm_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl")
}

# Load preprocessing tools
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load model performance metrics from CSV
metrics_df = pd.read_csv("model_performance_metrics.csv", index_col=0)
metrics_dict = metrics_df.to_dict(orient="index")  # Convert to dictionary for easy access

# Define feature names
FEATURE_NAMES = [
    "pm25", "pm10", "no", "no2", "nox", "nh3", "so2", "co",
    "o3", "benzene", "humidity", "wind_speed", "wind_direction",
    "solar_radiation", "rainfall", "air_temperature"
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Validate input
        if "features" not in data or not isinstance(data["features"], list) or len(data["features"]) != 16:
            return jsonify({"error": "Invalid input. Expected 16 feature values in a list."}), 400

        # Convert input to DataFrame
        df_input = pd.DataFrame([data["features"]], columns=FEATURE_NAMES)

        # Scale input
        X_scaled = scaler.transform(df_input)

        # Get predictions
        predictions = {}
        for model_name, model in models.items():
            y_pred = model.predict(X_scaled)
            y_pred_label = label_encoder.inverse_transform(y_pred)[0]
            predictions[model_name] = y_pred_label

        # Convert predictions to DataFrame for structured output
        predictions_df = pd.DataFrame(list(predictions.items()), columns=["Model", "Predicted Efficiency Category"])

        # Get the best-performing model based on Accuracy
        best_model = metrics_df["Accuracy"].idxmax()
        best_model_accuracy = metrics_df.loc[best_model, "Accuracy"]

        return jsonify({
            "predictions": predictions_df.to_dict(orient="records"),
            "metrics": metrics_dict,
            "final_recommendation": f"The best performing model is '{best_model}' with an accuracy of {best_model_accuracy:.2%}."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
