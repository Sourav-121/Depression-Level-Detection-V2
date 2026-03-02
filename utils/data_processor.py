import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def encode_categorical_features(input_df, reference_df):
    """
    Encode categorical features using the same encoding as training data
    """
    # Create a copy of input dataframe
    encoded_df = input_df.copy()
    
    # Define columns to encode
    categorical_cols = ['gender', 'ageGroup', 'universityType', 'department', 
                        'studyLevel', 'universityName']
    
    # Encode each categorical column
    for col in categorical_cols:
        if col in encoded_df.columns:
            # Combine reference and input for consistent encoding
            combined = pd.concat([reference_df[col], encoded_df[col]], axis=0)
            le = LabelEncoder()
            le.fit(combined)
            
            # Transform input
            encoded_df[col] = le.transform(encoded_df[col])
    
    # Encode symptom columns
    symptom_cols = [col for col in input_df.columns if col not in categorical_cols + ['Class']]
    
    # Define frequency mapping
    freq_mapping = {
        'Never': 0,
        'Rarely (less than one day)': 1,
        'Occasionally (1-2 days)': 2,
        'Frequently (3-4 days)': 3,
        'Most of the time (5-7 days)': 4
    }
    
    # Map symptom frequencies
    for col in symptom_cols:
        if col in encoded_df.columns:
            encoded_df[col] = encoded_df[col].map(freq_mapping).fillna(2)
    
    return encoded_df

def preprocess_input(input_data, reference_df):
    """
    Preprocess user input for prediction
    """
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode features
    encoded_df = encode_categorical_features(input_df, reference_df)
    
    return encoded_df

def calculate_severity_score(input_data):
    """
    Calculate a severity score based on symptom frequencies
    """
    # Define symptom weights (can be adjusted based on clinical importance)
    core_symptoms = ['persistentSadness', 'lostInterest', 'hopelessness', 
                    'selfHarm', 'futureHopelessness']
    
    # Frequency mapping for scoring
    freq_scores = {
        'Never': 0,
        'Rarely (less than one day)': 1,
        'Occasionally (1-2 days)': 2,
        'Frequently (3-4 days)': 3,
        'Most of the time (5-7 days)': 4
    }
    
    # Calculate total score
    total_score = 0
    max_possible = 0
    
    for symptom, value in input_data.items():
        if symptom not in ['gender', 'ageGroup', 'universityType', 'department', 
                          'studyLevel', 'universityName', 'Class']:
            score = freq_scores.get(value, 0)
            total_score += score
            max_possible += 4
    
    # Calculate percentage
    if max_possible > 0:
        severity_percentage = (total_score / max_possible) * 100
    else:
        severity_percentage = 0
    
    return severity_percentage, total_score