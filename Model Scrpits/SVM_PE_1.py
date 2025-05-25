# Import necessary libraries
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Start execution time
start_time = time.time()

# Load dataset
diabetes_dataset = pd.read_csv('/content/dataset.csv')

# Preprocess data
diabetes_dataset.drop(columns='gender', inplace=True)
diabetes_dataset['smoking_history'] = diabetes_dataset['smoking_history'].map({
    'never': 0, 'No Info': 1, 'former': 2, 'current': 3
})
diabetes_dataset.dropna(inplace=True)

# Split features and target
X = diabetes_dataset.drop(columns='diabetes')
Y = diabetes_dataset['diabetes']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model
classifier = LogisticRegression(max_iter=200)
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
pickle.dump(classifier, open('diabetes_model.sav', 'wb'))
print("Model saved successfully!")

# Feature correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(diabetes_dataset.corr(), annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.show()

# Class balance visualization
diabetes_dataset['diabetes'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'orange'], labels=['Non-Diabetic', 'Diabetic'], startangle=90, figsize=(6, 6))
plt.ylabel('')
plt.show()

# Feature importance visualization
feature_importance = pd.Series(classifier.coef_[0], index=X.columns).sort_values(ascending=False)
feature_importance.plot(kind='bar', color='teal', figsize=(10, 6))
plt.ylabel('Coefficient Value')
plt.grid(axis='y', linestyle='--')
plt.show()

# Confusion matrix
cm = confusion_matrix(Y_test, classifier.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Diabetic', 'Diabetic'])
disp.plot(cmap='Blues')

# Annotate confusion matrix
plt.text(0, 0.25, '\nTrue Negative', ha='center', va='center', color='white')
plt.text(1, 1.25, '\nTrue Positive', ha='center', va='center', color='black')
plt.text(0, 1.25, '\nFalse Positive', ha='center', va='center', color='black')
plt.text(1, 0.25, '\nFalse Negative', ha='center', va='center', color='black')

plt.show()

# ROC curve
Y_test_probs = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(Y_test, Y_test_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Misclassification analysis
misclassified = X_test[classifier.predict(X_test) != Y_test]
plt.scatter(misclassified['age'], misclassified['bmi'], color='red', label='Misclassified')
plt.scatter(X_test['age'], X_test['bmi'], alpha=0.3, label='All Data', color='gray')
plt.title('Misclassification Analysis')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.legend()
plt.grid()
plt.show()
