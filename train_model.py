import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from utils import custom_tokenizer

# Define a custom tokenizer function
def custom_tokenizer(text):
  return text.split(',')

# Load dataset
data = pd.read_csv('Symptom2Disease.csv')

# Data Cleaning: Handle missing values and remove duplicates
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Features (Symptoms) and target (Disease) extraction
X = data['text']  # Symptoms text
y = data['label']  # Disease labels

# Text Vectorization: Convert symptoms text to numerical features
vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, token_pattern=None)
X = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Balancing: Use SMOTE to address class imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Model Tuning: Hyperparameter tuning using Grid Search
param_grid = {
  'n_estimators': [100, 200, 300],
  'max_depth': [None, 10, 20, 30],
  'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model from Grid Search
best_model = grid_search.best_estimator_

# Train the model
best_model.fit(X_train, y_train)

# Evaluate the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Print confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model and vectorizer using pickle
try:
  with open('disease_prediction_model.pickle', 'wb') as model_file:
    pickle.dump(best_model, model_file)
    print('Model saved successfully as disease_prediction_model.pickle')

  with open('vectorizer.pickle', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
    print('Vectorizer saved successfully as vectorizer.pickle')
except Exception as e:
  print(f"An error occurred while saving the model: {e}")