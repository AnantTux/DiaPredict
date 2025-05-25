# Import necessary libraries
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Start execution time
start_time = time.time()

# Load dataset
diabetes_dataset = pd.read_csv('/content/dataset.csv')

# Drop unnecessary columns and map categorical data
diabetes_dataset.drop(columns='gender', inplace=True)
diabetes_dataset['smoking_history'] = diabetes_dataset['smoking_history'].map({
    'never': 0, 'No Info': 1, 'former': 2, 'current': 3
})
diabetes_dataset.dropna(inplace=True)

# Split features and target
X = diabetes_dataset.drop(columns='diabetes')
Y = diabetes_dataset['diabetes']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Train model
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, Y_train)

# Evaluate model
training_data_accuracy = accuracy_score(classifier.predict(X_train), Y_train)
test_data_accuracy = accuracy_score(classifier.predict(X_test), Y_test)

print(f'Training Accuracy: {training_data_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_data_accuracy * 100:.2f}%')

# Execution time
execution_time = time.time() - start_time
print(f"Total execution time: {execution_time:.4f} seconds")

# Accuracy plot
plt.figure(figsize=(8, 5))
plt.bar(['Training', 'Test'], [training_data_accuracy * 100, test_data_accuracy * 100], color=['blue', 'orange'])
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Model Performance')
plt.grid(axis='y', linestyle='--')
plt.show()

# Prediction system
input_data = np.asarray((80, 0, 1, 0, 25.19, 6.6, 140)).reshape(1, -1)
prediction = classifier.predict(input_data)
print('Diabetic' if prediction[0] == 1 else 'Not Diabetic')

# Save model
pickle.dump(classifier, open('diabetes_decision_tree_model.sav', 'wb'))
print("Model saved successfully!")