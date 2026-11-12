import pandas as pd
import os
import pickle
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

#LOAD DATASET
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

print("Dataset Path:", path)

csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
csv_file_path = os.path.join(path, csv_file)

df = pd.read_csv(csv_file_path)

#SELECT COLUMNS
columns_to_keep = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'Contract',
    'TotalCharges', 'Churn'
]

df = df[columns_to_keep]

#Data preprocessing

df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'], errors='coerce'
)

df['TotalCharges'] = df['TotalCharges'].fillna(
    df['TotalCharges'].mean()
)

#LABEL ENCODING
encoders = {}

categorical_columns = [
    'gender', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines',
    'Contract', 'Churn'
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

#SPLIT FEATURES AND TARGET
X = df.drop('Churn', axis=1)
y = df['Churn']

# FEATURE SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

#MODEL TRAINING
model = LogisticRegression(
    C=0.1,
    solver='liblinear',
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

#MODEL EVALUATION
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#SAVE MODEL SCALER AND ENCODERS
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(encoders, open('encoders.pkl', 'wb'))

print("\nModel, Scaler and Encoders saved successfully!")


