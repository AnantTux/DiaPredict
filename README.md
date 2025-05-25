# 🩺 Diabetes Prediction System

This Flask-based web application predicts the likelihood of diabetes using **multiple machine learning models**, including **Decision Tree, KNN, Naive Bayes, Random Forest, Logistic Regression, and SVM**.

## 🚀 Features
- ✅ Predicts diabetes risk using multiple ML models  
- ✅ User-friendly **web interface** for entering health data  
- ✅ **Preloaded models** for fast execution  
- ✅ Stylish **dark-themed UI** inspired by modern web designs  

---

## 🏗️ Installation & Setup

### 📌 **1. Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/diabetes-prediction.git
cd diabetes-prediction
```

### 📌 **2. Install Dependencies**
Ensure you have **Python 3.x** installed, then install the required libraries:
```bash
pip install flask numpy pickle scikit-learn
```

### 📌 **3. Place Model Files**
Ensure all **ML model files** (`.sav`) are inside the `models/` directory:
```plaintext
diabetes-prediction/
│── models/
│   ├── diabetes_decision_tree_model.sav
│   ├── diabetes_knn_model.sav
│   ├── diabetes_naive_bayes_model.sav
│   ├── diabetes_rf_model.sav
│   ├── diabetes_logistic_regression_model.sav
│   ├── diabetes_svm_model.sav
│── templates/
│   ├── index.html
│── static/
│   ├── styles.css
│── app.py
│── requirements.txt
│── README.md
```

---

## 🔥 Running the Flask App

### 📌 **1. Start the Flask Server**
Run the following command:
```bash
python app.py
```
The click on the link in terminal for opening the local Server
```
http://{ip}/
```

### 📌 **2. Optionally Expose via Ngrok**
If you want to make your app accessible **publicly**, install and run Ngrok:
```bash
ngrok http 5000
```
Use the generated **public URL** to access the app from anywhere!

---

## ✨ Usage Guide

### **Step 1:** Open the Web Interface
Go to the browser in which the link is opened and enter the required health details:
- **Age**
- **Smoking History** (`Never`, `No Info`, `Former Smoker`, `Current Smoker`)
- **Hypertension** (`Yes` or `No`)
- **Heart Disease** (`Yes` or `No`)
- **BMI** (Body Mass Index)
- **HbA1c Level** (Blood Sugar Indicator)
- **Glucose Level** (mg/dL)

### **Step 2:** Click "Predict"
The app will analyze the data using **multiple machine learning models** and display the predictions.

### **Step 3:** View Results
Each model’s prediction (`Diabetic` or `Not Diabetic`) will be listed in the results section.

---

## 🛠️ Troubleshooting

🔹 **Flask not starting?**  
Ensure dependencies are installed correctly:
```bash
pip install -r requirements.txt
```

🔹 **Ngrok not working?**  
Verify Ngrok installation by running:
```bash
ngrok --version
```
Set up your **authtoken** if needed:
```bash
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

🔹 **Model file missing?**  
Make sure `.sav` model files exist in the `models/` directory.

---

## 🏆 Credits
- **Machine Learning Models**: Developed using `scikit-learn`
- **UI Design**: Dark-mode **modern aesthetics**
- **Backend**: Flask-based **efficient prediction system**

---

## 📜 License
This project is **open-source** under the MIT License. Feel free to contribute and improve!

---

🚀 **Built for accurate diabetes risk assessment using machine learning!**  
Contributions & feedback are welcome! 🤖🔥

![VIT Bhopal](https://img.shields.io/badge/VIT-Bhopal-blue)
![CSS](https://img.shields.io/badge/CSS-Styles-orange)
![HTML](https://img.shields.io/badge/HTML-Markup-red)
![Python](https://img.shields.io/badge/Python-Programming-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-AI-brightgreen)
![Diabetes](https://img.shields.io/badge/Diabetes-Prediction-purple)
![Ngrok](https://img.shields.io/badge/Ngrok-Tunneling-black)
![MIT License](https://img.shields.io/badge/License-MIT-green)


