
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("lung_cancer_model.pkl", "rb"))
scaler = pickle.load(open("lung_cancer_scaler.pkl", "rb"))

@app.route("/")
def home():
    return "Lung Cancer Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        features = [data[key] for key in sorted(data.keys())]
        scaled = scaler.transform([features])
        prediction = model.predict(scaled)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
