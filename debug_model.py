import joblib
import pandas as pd

# Load the model
model = joblib.load('Logistic_Regression.joblib')

# Check what features the model expects
if hasattr(model, 'feature_names_in_'):
    print("Features the model expects:")
    for i, feature in enumerate(model.feature_names_in_):
        print(f"{i+1}. {feature}")
    print(f"\nTotal features: {len(model.feature_names_in_)}")
else:
    print("Model doesn't have feature_names_in_ attribute")
    
    # Try to get from the first step if it's a pipeline
    if hasattr(model, 'steps'):
        print("\nModel is a pipeline with steps:")
        for step_name, step_obj in model.steps:
            print(f"- {step_name}: {type(step_obj).__name__}")
            if hasattr(step_obj, 'feature_names_in_'):
                print(f"  Features in this step: {step_obj.feature_names_in_}")