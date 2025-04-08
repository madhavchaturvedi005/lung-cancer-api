from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("lung_cancer_model.pkl", "rb"))
scaler = pickle.load(open("lung_cancer_scaler.pkl", "rb"))

@app.route('/')
def home():
    return "Lung Cancer Prediction API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Assuming input features are in the correct order
        features = [
            data["GENDER"],
            data["AGE"],
            data["SMOKING"],
            data["YELLOW_FINGERS"],
            data["ANXIETY"],
            data["PEER_PRESSURE"],
            data["CHRONIC_DISEASE"],
            data["FATIGUE"],
            data["ALLERGY"],
            data["WHEEZING"],
            data["ALCOHOL_CONSUMING"],
            data["COUGHING"],
            data["SHORTNESS_OF_BREATH"],
            data["SWALLOWING_DIFFICULTY"],
            data["CHEST_PAIN"]
        ]

        features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
