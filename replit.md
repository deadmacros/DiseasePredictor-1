# Disease Prediction System

## Overview

This is a Flask-based web application that provides AI-powered disease prediction based on blood test parameters. The system uses a machine learning model to analyze six key blood parameters (Hemoglobin, RBC, WBC, Platelet, ALT, Creatinine) and predicts potential health conditions including Healthy status, Kidney Disease, Liver Disease, and Leukemia. The application features a responsive dark-themed web interface for medical professionals and patients to input blood test results and receive instant predictions with confidence scores.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: HTML templates with Bootstrap 5 dark theme
- **Styling**: Custom CSS with medical color scheme using CSS variables
- **UI Components**: Responsive cards, forms, progress bars, and badges
- **Theme**: Dark theme optimized for medical environments
- **Icons**: Bootstrap Icons for medical and UI elements

### Backend Architecture
- **Framework**: Flask web framework with Python
- **Model Architecture**: RandomForest classifier for multi-class disease prediction
- **Data Processing**: NumPy and Pandas for data manipulation
- **Model Persistence**: Joblib for saving/loading trained models
- **Validation**: Custom input validation against medical parameter ranges

### Machine Learning Pipeline
- **Algorithm**: RandomForest classifier with StandardScaler preprocessing
- **Training Data**: Synthetic medical data generation with realistic parameter ranges
- **Features**: Six blood parameters with medically accurate normal ranges
- **Classes**: Four prediction classes (Healthy, Kidney Disease, Liver Disease, Leukemia)
- **Model Storage**: Automatic model training on first run, persistent storage for subsequent uses

### Application Structure
- **Entry Point**: main.py serves the Flask application
- **Core Logic**: app.py contains routes, model loading, and validation
- **ML Module**: model_trainer.py handles data generation and model training
- **Templates**: Separate HTML templates for index, results, and about pages
- **Static Assets**: Custom CSS for medical theme styling

### Data Architecture
- **Parameter Ranges**: Hardcoded medical reference ranges for validation
- **Synthetic Data**: Programmatically generated realistic blood test data
- **Feature Engineering**: StandardScaler normalization for model input
- **Prediction Output**: Multi-class predictions with confidence scores

## External Dependencies

### Python Libraries
- **Flask**: Web framework for HTTP routing and template rendering
- **scikit-learn**: Machine learning library for RandomForest classifier and preprocessing
- **NumPy**: Numerical computing for array operations and data generation
- **Pandas**: Data manipulation and analysis framework
- **Joblib**: Model serialization and persistence

### Frontend Dependencies
- **Bootstrap 5**: CSS framework with dark theme from cdn.replit.com
- **Bootstrap Icons**: Icon library from cdn.jsdelivr.net

### Runtime Environment
- **Python Runtime**: Requires Python with scientific computing stack
- **File System**: Local file system for model persistence (disease_prediction_model.pkl, feature_columns.pkl)
- **Environment Variables**: SESSION_SECRET for Flask session management