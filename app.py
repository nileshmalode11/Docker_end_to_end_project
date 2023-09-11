import json

import pickle

from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open("linear_model1.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html') 

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]
    print(data)
    # Convert the received JSON data into a NumPy array and reshape it
    data_array = np.array(list(data.values())).reshape(1, -1)
    output = model.predict(data_array)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form values from the request
    data = [float(request.form[key]) for key in request.form.keys()]
    data = np.array(data).reshape(1, -1)
    output = model.predict(data)
    return render_template("home.html", prediction_text="The House price is {}".format(output[0]))

if __name__ == '__main__':
    app.run(debug=True)
    