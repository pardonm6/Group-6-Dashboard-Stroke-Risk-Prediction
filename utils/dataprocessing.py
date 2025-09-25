"""
Data Processing Utilities for Stroke Risk Prediction Dashboard
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

class DataProcessor:
    """Class for processing stroke risk prediction data"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        df_clean = df.copy()
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        print(f"Data cleaned: {df_clean.shape[0]} rows remaining")
        return df_clean
    
    def encode_categorical_features(self, df, categorical_columns):
        """Encode categorical features"""
        df_encoded = df.copy()
        
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df_encoded[f'{column}_encoded'] = self.label_encoders[column].fit_transform(df[column])
            else:
                df_encoded[f'{column}_encoded'] = self.label_encoders[column].transform(df[column])
        
        return df_encoded
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def prepare_features(self, df, target_column='stroke'):
        """Prepare features and target for modeling"""
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Identify categorical columns
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        # Encode categorical features
        if categorical_columns:
            X = self.encode_categorical_features(X, categorical_columns)
            # Drop original categorical columns
            X = X.drop(columns=categorical_columns)
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def save_preprocessor(self, filepath):
        """Save the preprocessor object"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'scaler': self.scaler
            }, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """Load the preprocessor object"""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
            self.label_encoders = preprocessor['label_encoders']
            self.scaler = preprocessor['scaler']
        print(f"Preprocessor loaded from {filepath}")

def calculate_risk_factors(df):
    """Calculate additional risk factors from the data"""
    risk_factors = {}
    
    # Age risk factor
    if 'age' in df.columns:
        risk_factors['age_risk'] = pd.cut(df['age'], 
                                          bins=[0, 45, 65, 100], 
                                          labels=['Low', 'Medium', 'High'])
    
    # BMI risk factor
    if 'bmi' in df.columns:
        risk_factors['bmi_category'] = pd.cut(df['bmi'], 
                                              bins=[0, 18.5, 25, 30, 100], 
                                              labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Glucose risk factor
    if 'avg_glucose_level' in df.columns:
        risk_factors['glucose_category'] = pd.cut(df['avg_glucose_level'], 
                                                   bins=[0, 100, 140, 200, 400], 
                                                   labels=['Normal', 'Prediabetes', 'Diabetes', 'High'])
    
    return pd.DataFrame(risk_factors)

def generate_summary_statistics(df):
    """Generate summary statistics for the dataset"""
    summary = {
        'total_patients': len(df),
        'stroke_cases': df['stroke'].sum() if 'stroke' in df.columns else 0,
        'stroke_rate': df['stroke'].mean() * 100 if 'stroke' in df.columns else 0,
        'avg_age': df['age'].mean() if 'age' in df.columns else 0,
        'avg_bmi': df['bmi'].mean() if 'bmi' in df.columns else 0,
        'avg_glucose': df['avg_glucose_level'].mean() if 'avg_glucose_level' in df.columns else 0,
        'hypertension_rate': df['hypertension'].mean() * 100 if 'hypertension' in df.columns else 0,
        'heart_disease_rate': df['heart_disease'].mean() * 100 if 'heart_disease' in df.columns else 0
    }
    
    return summary

def validate_input(patient_data, required_fields):
    """Validate patient input data"""
    errors = []
    
    for field in required_fields:
        if field not in patient_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate age
    if 'age' in patient_data:
        if patient_data['age'] < 0 or patient_data['age'] > 120:
            errors.append("Age must be between 0 and 120")
    
    # Validate BMI
    if 'bmi' in patient_data:
        if patient_data['bmi'] < 10 or patient_data['bmi'] > 60:
            errors.append("BMI must be between 10 and 60")
    
    # Validate glucose
    if 'avg_glucose_level' in patient_data:
        if patient_data['avg_glucose_level'] < 50 or patient_data['avg_glucose_level'] > 400:
            errors.append("Glucose level must be between 50 and 400")
    
    return errors

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = DataProcessor()
    
    # Example: Load and process data
    # data = processor.load_data('path/to/data.csv')
    # if data is not None:
    #     clean_data = processor.clean_data(data)
    #     X, y = processor.prepare_features(clean_data)
    #     X_train, X_test, y_train, y_test = processor.split_data(X, y)
    
    print("Data processing utilities loaded successfully")