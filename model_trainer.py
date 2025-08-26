import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import logging

def create_synthetic_data(n_samples=5000):
    """
    Create synthetic medical data with realistic blood parameter values
    """
    np.random.seed(42)  # For reproducibility
    
    data = []
    
    # Define realistic parameter ranges for each condition
    conditions = {
        'Healthy': {
            'Hemoglobin': (12.0, 16.0),
            'RBC': (4.2, 5.4),
            'WBC': (4000, 11000),
            'Platelet': (150000, 450000),
            'ALT': (7, 56),
            'Creatinine': (0.7, 1.3)
        },
        'Kidney Disease': {
            'Hemoglobin': (8.0, 12.0),  # Lower due to reduced EPO
            'RBC': (3.0, 4.2),
            'WBC': (4000, 15000),
            'Platelet': (100000, 400000),
            'ALT': (10, 80),
            'Creatinine': (2.0, 8.0)  # Elevated
        },
        'Liver Disease': {
            'Hemoglobin': (9.0, 13.0),
            'RBC': (3.5, 4.8),
            'WBC': (3000, 12000),
            'Platelet': (80000, 200000),  # Reduced due to hypersplenism
            'ALT': (80, 200),  # Elevated
            'Creatinine': (0.5, 2.5)
        },
        'Leukemia': {
            'Hemoglobin': (6.0, 10.0),  # Severe anemia
            'RBC': (2.5, 3.5),
            'WBC': (15000, 50000),  # Very high
            'Platelet': (20000, 100000),  # Very low
            'ALT': (20, 100),
            'Creatinine': (0.8, 2.0)
        },
        'Anemia': {
            'Hemoglobin': (6.5, 10.0),  # Low
            'RBC': (2.8, 3.8),  # Low
            'WBC': (3500, 9000),
            'Platelet': (120000, 350000),
            'ALT': (15, 70),
            'Creatinine': (0.6, 1.5)
        }
    }
    
    samples_per_condition = n_samples // len(conditions)
    
    for condition, ranges in conditions.items():
        for _ in range(samples_per_condition):
            sample = {}
            
            # Generate values with some variation and occasional outliers
            for param, (min_val, max_val) in ranges.items():
                # Add some noise and occasional outliers
                if np.random.random() < 0.05:  # 5% chance of outlier
                    # Create outliers within reasonable medical bounds
                    if param == 'Hemoglobin':
                        value = np.random.normal((min_val + max_val) / 2, (max_val - min_val) / 3)
                        value = np.clip(value, 6.0, 20.0)
                    elif param == 'RBC':
                        value = np.random.normal((min_val + max_val) / 2, (max_val - min_val) / 3)
                        value = np.clip(value, 2.0, 8.0)
                    elif param == 'WBC':
                        value = np.random.normal((min_val + max_val) / 2, (max_val - min_val) / 4)
                        value = np.clip(value, 1000, 50000)
                    elif param == 'Platelet':
                        value = np.random.normal((min_val + max_val) / 2, (max_val - min_val) / 4)
                        value = np.clip(value, 50000, 800000)
                    elif param == 'ALT':
                        value = np.random.normal((min_val + max_val) / 2, (max_val - min_val) / 3)
                        value = np.clip(value, 5, 200)
                    elif param == 'Creatinine':
                        value = np.random.normal((min_val + max_val) / 2, (max_val - min_val) / 3)
                        value = np.clip(value, 0.3, 10.0)
                else:
                    # Normal distribution within the range
                    value = np.random.uniform(min_val, max_val)
                
                # Round appropriately
                if param in ['WBC', 'Platelet']:
                    sample[param] = round(value)
                else:
                    sample[param] = round(value, 1)
            
            sample['Disease'] = condition
            data.append(sample)
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    logging.info(f"Created synthetic dataset with {len(df)} samples")
    logging.info(f"Disease distribution:\n{df['Disease'].value_counts()}")
    
    return df

def train_model(data):
    """
    Train a RandomForestClassifier on the synthetic data
    """
    # Separate features and target
    feature_columns = ['Hemoglobin', 'RBC', 'WBC', 'Platelet', 'ALT', 'Creatinine']
    X = data[feature_columns]
    y = data['Disease']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logging.info(f"Model trained with accuracy: {accuracy:.4f}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logging.info(f"Feature Importance:\n{feature_importance}")
    
    return rf_model, feature_columns

if __name__ == "__main__":
    # Create and train model for testing
    logging.basicConfig(level=logging.INFO)
    
    print("Creating synthetic data...")
    data = create_synthetic_data(5000)
    
    print("Training model...")
    model, features = train_model(data)
    
    print("Model training completed!")
