"""""
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained RandomForest model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Route for rendering the main HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions via API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    
    # Convert data into the format required for prediction
    features = np.array(data['features']).reshape(1, -1)  # Reshape for a single prediction
    
    # Make prediction using the model
    prediction = model.predict(features)
    
    # Send the prediction result
    return jsonify({'prediction': round(prediction[0], 2)})

if __name__ == '__main__':
    app.run(debug=True)"""

from flask import Flask, request, jsonify
from model import predict_diamond_price  # Import the prediction function from model.py

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']

    # Extract features from the request
    carat = features['carat']
    cut = features['cut']
    color = features['color']
    clarity = features['clarity']
    depth = features['depth']
    table = features['table']
    volume = features['volume']

    # Use the predict_diamond_price function to get the prediction
    predicted_price = predict_diamond_price(carat, cut, color, clarity, depth, table, volume)

    return jsonify({'prediction': round(predicted_price, 2)})

if __name__ == '__main__':
    app.run(debug=True)

