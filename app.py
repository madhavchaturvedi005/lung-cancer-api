@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        features = [data[key] for key in sorted(data.keys())]
        scaled = scaler.transform([features])
        prediction = model.predict(scaled)
        probability = model.predict_proba(scaled)[0][1]  # probability of class 1 (lung cancer)
        return jsonify({
            "prediction": int(prediction[0]),
            "probability_of_cancer": round(float(probability), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)})
