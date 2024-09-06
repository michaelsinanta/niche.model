import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("model/roles-recommendation/model.pkl", "rb"))
scaler = pickle.load(open("model/roles-recommendation/scaler.pkl", "rb"))


@app.route("/predict-role", methods=["POST"])
def predict():

    data = request.get_json()

    new_sample = np.array(data["features"]).reshape(1, -1)

    new_sample_scaled = scaler.transform(new_sample)

    new_pred = model.predict(new_sample_scaled)

    return jsonify({"predicted_role": new_pred[0]})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
