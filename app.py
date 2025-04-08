from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Initialize the Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Lung Cancer Prediction API is working!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Make sure keys are sorted as per training
        features = [data[key] for key in sorted(data.keys())]

        # Apply the same preprocessing (scaling) used during training
        scaled = scaler.transform([features])

        # Make prediction
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]  # probability of class 1 (lung cancer)

        return jsonify({
            "prediction": int(prediction),
            "probability_of_cancer": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app locally (for testing)
if __name__ == "__main__":
    app.run(debug=True)
