from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Liver Cirrhosis Stage Detection API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    return jsonify({"stage": int(prediction[0])})

if __name__ == "__main__":
    app.run()
