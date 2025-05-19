from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load('student_model.pkl')


@app.route('/')
def home():
    return "🎯 Student Performance Analyzer API is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        gender = int(data['gender'])
        study_time = float(data['study_time'])
        attendance = float(data['attendance'])
        previous_marks = float(data['previous_marks'])
        behavior_score = int(data['behavior_score'])
        internet_usage = int(data['internet_usage'])

        input_data = np.array([[gender, study_time, attendance, previous_marks, behavior_score, internet_usage]])
        prediction = model.predict(input_data)[0]
        label_map = {0: "Poor", 1: "Average", 2: "Excellent"}

        return jsonify({'performance': label_map[prediction]})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict-bulk', methods=['POST'])
def predict_bulk():
    file = request.files['file']
    
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format. Please upload a CSV or Excel file.'}), 400

        required_cols = ['gender', 'study_time', 'attendance', 'previous_marks', 'behavior_score', 'internet_usage']
        if not all(col in df.columns for col in required_cols):
            return jsonify({'error': f'Missing required columns: {required_cols}'}), 400

        predictions = model.predict(df[required_cols])
        label_map = {0: "Poor", 1: "Average", 2: "Excellent"}
        df['Predicted Performance'] = [label_map[p] for p in predictions]

        return df.to_dict(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
