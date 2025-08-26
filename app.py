import os
import logging
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for
import joblib
from model_trainer import create_synthetic_data, train_model

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Medical parameter reference ranges
PARAMETER_RANGES = {
    'Hemoglobin': {'min': 6.0, 'max': 20.0, 'normal': '12.0-16.0 g/dL', 'unit': 'g/dL'},
    'RBC': {'min': 2.0, 'max': 8.0, 'normal': '4.2-5.4 million/μL', 'unit': 'million/μL'},
    'WBC': {'min': 1000, 'max': 50000, 'normal': '4000-11000 /μL', 'unit': '/μL'},
    'Platelet': {'min': 50000, 'max': 800000, 'normal': '150000-450000 /μL', 'unit': '/μL'},
    'ALT': {'min': 5, 'max': 200, 'normal': '7-56 U/L', 'unit': 'U/L'},
    'Creatinine': {'min': 0.3, 'max': 10.0, 'normal': '0.7-1.3 mg/dL', 'unit': 'mg/dL'}
}

def load_or_create_model():
    """Load existing model or create and train a new one"""
    try:
        model = joblib.load('disease_prediction_model.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        logging.info("Loaded existing model")
        return model, feature_columns
    except FileNotFoundError:
        logging.info("No existing model found. Creating and training new model...")
        # Create synthetic data and train model
        data = create_synthetic_data(n_samples=5000)
        model, feature_columns = train_model(data)
        
        # Save the model and feature columns
        joblib.dump(model, 'disease_prediction_model.pkl')
        joblib.dump(feature_columns, 'feature_columns.pkl')
        logging.info("Model trained and saved successfully")
        return model, feature_columns

# Load or create the model at startup
model, feature_columns = load_or_create_model()

def validate_input(value, param_name):
    """Validate input parameters against medical ranges"""
    try:
        value = float(value)
        param_range = PARAMETER_RANGES[param_name]
        if value < param_range['min'] or value > param_range['max']:
            return False, f"{param_name} must be between {param_range['min']} and {param_range['max']} {param_range['unit']}"
        return True, value
    except ValueError:
        return False, f"{param_name} must be a valid number"

@app.route('/')
def index():
    """Main page with input form"""
    return render_template('index.html', parameter_ranges=PARAMETER_RANGES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        form_data = {}
        errors = []
        
        for param in ['Hemoglobin', 'RBC', 'WBC', 'Platelet', 'ALT', 'Creatinine']:
            value = request.form.get(param, '').strip()
            if not value:
                errors.append(f"{param} is required")
                continue
            
            is_valid, result = validate_input(value, param)
            if not is_valid:
                errors.append(result)
            else:
                form_data[param] = result
        
        if errors:
            for error in errors:
                flash(error, 'danger')
            return redirect(url_for('index'))
        
        # Create input array for prediction
        input_data = np.array([[form_data[col] for col in feature_columns]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Get confidence scores for all classes
        class_names = model.classes_
        confidence_scores = {}
        for i, class_name in enumerate(class_names):
            confidence_scores[class_name] = round(prediction_proba[i] * 100, 2)
        
        # Sort confidence scores in descending order
        sorted_scores = dict(sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True))
        
        # Determine risk level based on prediction
        risk_level = "Low" if prediction == "Healthy" else "High"
        risk_color = "success" if prediction == "Healthy" else "danger"
        
        return render_template('result.html', 
                             prediction=prediction,
                             confidence_scores=sorted_scores,
                             input_data=form_data,
                             parameter_ranges=PARAMETER_RANGES,
                             risk_level=risk_level,
                             risk_color=risk_color)
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        flash('An error occurred during prediction. Please try again.', 'danger')
        return redirect(url_for('index'))

@app.route('/about')
def about():
    """About page with information about the model"""
    model_info = {
        'algorithm': 'Random Forest Classifier',
        'features': feature_columns,
        'classes': list(model.classes_),
        'accuracy': 'Trained on synthetic medical data'
    }
    return render_template('about.html', model_info=model_info, parameter_ranges=PARAMETER_RANGES)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
