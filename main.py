import joblib
import numpy as np
import warnings

warnings.filterwarnings('ignore')

svm = joblib.load('SVM.joblib')
scaler = joblib.load('scaler.joblib')

def make_predictions():
    feature_values = []
    features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate','Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    for feature in features:
        feature_value = float(input(f'Enter the value of {feature}  '))
        feature_values.append(feature_value)
        
    feature_values = np.array(feature_values)
    feature_values = feature_values.reshape(1,-1)
    feature_values = scaler.transform(feature_values)
    prediction = svm.predict(feature_values)
    print("=========================================")
    if prediction == 0:
        print("Water is not safe for human consumption")
    else:
        print("Water is safe for human consumption")

make_predictions()