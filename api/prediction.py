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
import os

app = Flask(__name__)

current_directory = os.path.split(os.getcwd())[0]


def load_model(model_type):
    if model_type == 'svm':
        print("svm")
        file_path = os.path.join(current_directory, 'models', 'M23CSA009_gamma:0.01_C:1.joblib')
    elif model_type == 'logistic':
        file_path = os.path.join(current_directory, 'models', 'M23CSA009_lr_lbfgs.joblib')
    elif model_type == 'decision_tree':
        file_path = os.path.join(current_directory, 'models', 'tree_max_depth:50.joblib')
    else:
        raise ValueError("Invalid model_type. Choose 'svm', 'logistic', or 'decision_tree'.")
    
    model = load(file_path)
    return model

model = load_model('svm')

@app.route('/predict/<model_type>', methods=['POST'])
def compare_digits():
    try:
        # Get the two image files from the request
        data = request.get_json()  # Parse JSON data from the request body
        image1 = data.get('image1', [])
        image2 = data.get('image2', [])

        # Preprocess the images and make predictions
        digit1 = predict_digit(image1)
        digit2 = predict_digit(image2)

        # Compare the predicted digits and return the result
        if (digit1 == digit2):
            return jsonify({'Result':"Images same"})

        else:
            return jsonify({'Result':"Images different"})
    except Exception as e:
        return jsonify({'ERROR': str(e)})
    
def predict_digit(image):
    try:
        # Convert the input list to a numpy array and preprocess for prediction
        reshaped_image = np.array(image, dtype=np.float32).reshape(1, 28, 28, 1) / 255

        prediction = model.predict(reshaped_image)
        digit = np.argmax(prediction)

        return digit
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run()