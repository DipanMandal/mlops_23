import matplotlib.pyplot as plt
import sys
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import pdb
from joblib import dump,load
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)

model = load('./models/svm_gamma:0.01_C:1.joblib')

@app.route('/predict', methods=['POST'])
def pred_model():
    data = request.get_json()
    image = data['image']
    pred1 = model.predict(image)
    #reurn pred1 in json
    return jsonify(prediction=pred1.tolist())


if __name__ == '__main__':
    app.run()