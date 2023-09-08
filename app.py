import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
##load the model
model=pickle.load(open("linear_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html') 

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data=request.jeson["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(data)
    print(output[0])
    return jesonify(output[0])


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(request.form.values())]
    output = model.predict(data)
    return render_template("home.html",predtiction_test="The House price is{}".format(output))

if __name__=='__main__':
    app.run(debug=True)
    