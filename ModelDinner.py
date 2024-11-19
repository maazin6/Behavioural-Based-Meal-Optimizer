import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
data = pd.read_csv('dinner.csv')

# Encode categorical columns
label_encoders = {}
for col in ['age', 'gender', 'region', 'meal_type', 'mood', 'allergies', 'meal_name']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split features and target variable
X = data.drop('meal_name', axis=1)
y = data['meal_name']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Hyperparameter grid for optimization
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit GridSearch to the data
grid_search.fit(X_train, y_train)

# Best parameters and estimator
best_rf = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)

# Predict on test set
y_pred = best_rf.predict(X_test)

# Model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(best_rf, 'dinner_model.pkl')
# Save the label encoders
joblib.dump(label_encoders, 'dinner_label_encoders.pkl')
