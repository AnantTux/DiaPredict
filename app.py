from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

model_files = {
    "Decision Tree": 'models/diabetes_decision_tree_model.sav',
    "KNN": 'models/diabetes_knn_model.sav',
    "Naive Bayes": 'models/diabetes_naive_bayes_model.sav',
    "Random Forest": 'models/diabetes_rf_model.sav',
    "Logistic Regression": 'models/diabetes_logistic_regression_model.sav',
    "SVM": 'models/diabetes_svm_model.sav'
}

def load_models():
    available_models = {}
    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    available_models[model_name] = pickle.load(f)
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
        else:
            print(f"Warning: {model_name} model file '{file_path}' not found!")
    return available_models

models = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        required_fields = [
            'age',
            'smoking_history',
            'hypertension',
            'heart_disease',
            'bmi',
            'hba1c_level',
            'glucose_level'
        ]

        input_data = []
        for field in required_fields:
            if field not in request.form:
                return jsonify({"error": f"Missing field: {field}"}), 400
            try:
                value = float(request.form[field])
                input_data.append(value)
            except ValueError:
                return jsonify({"error": f"Invalid value for {field}. Must be a number."}), 400

        input_array = np.asarray(input_data).reshape(1, -1)

        print("Received data shape:", input_array.shape)

        if input_array.shape[1] != 7:
            return jsonify({"error": "Incorrect number of input features. Expected 7."}), 400

        if not models:
            return jsonify({"error": "No models available for prediction"}), 500

        predictions = {
            name: "Diabetic" if model.predict(input_array)[0] == 1 else "Not Diabetic"
            for name, model in models.items()
        }

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
