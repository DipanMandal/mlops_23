from flask import Flask, request 

app = Flask(__name__)

@app.route("/hello/<val>")
def hello_world(val):
    return "<p>Hello, World!</p>" + val 


@app.route("/model", methods=['POST'])
def digit_model():
    js = request.get_json()
    x = js['x']
    y = js['y']

    return x+y