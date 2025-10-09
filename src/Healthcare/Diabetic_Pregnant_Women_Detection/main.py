import pickle
import pandas as pd

# Load the trained model
with open('models/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to take input and predict
def predict_diabetes(data):
    """
    data: dict containing all feature values
    Example:
    {
        'Pregnancies': 2,
        'Glucose': 120,
        'BloodPressure': 70,
        'SkinThickness': 20,
        'Insulin': 85,
        'BMI': 28.5,
        'DiabetesPedigreeFunction': 0.35,
        'Age': 29
    }
    """
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    if prediction == 1:
        return "The woman is likely Diabetic"
    else:
        return "The woman is Not Diabetic"

# Example test run
if __name__ == "__main__":
    # Example sample data
    sample_data = {
        'Pregnancies': 3,
        'Glucose': 130,
        'BloodPressure': 72,
        'SkinThickness': 28,
        'Insulin': 90,
        'BMI': 31.5,
        'DiabetesPedigreeFunction': 0.45,
        'Age': 33
    }

    result = predict_diabetes(sample_data)
    print(result)
