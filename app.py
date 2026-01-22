from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# Load model and column names
MODEL_PATH = 'models/price_model.pkl'
COLUMNS_PATH = 'models/model_columns.pkl'

model = None
model_columns = None

if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
else:
    print("Warning: Model files not found. Please run training script.")

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # input validation
        required_fields = ['Size', 'Bedrooms', 'Bathrooms', 'Age', 'Location']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
                
        # Create dataframe
        input_df = pd.DataFrame([data])
        
        # Preprocess (One-hot encoding)
        input_df = pd.get_dummies(input_df, columns=['Location'])
        
        # Ensure all columns exist
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Reorder columns
        input_df = input_df[model_columns]
        
        # Predict
        prediction = model.predict(input_df)[0]
        
        return jsonify({'price': float(prediction)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
