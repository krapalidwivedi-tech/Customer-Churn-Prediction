# Customer Churn Prediction using Machine Learning

## Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a telecom service or continue using it.

Customer churn is an important problem for telecom companies because losing existing customers can affect business growth and revenue.

This project uses customer demographic information, service details, contract information, and billing data to predict customer churn using Logistic Regression.

The model is trained on the Telco Customer Churn dataset and uses data preprocessing, feature encoding, and feature scaling to prepare the data for prediction.

## Objectives

- Understand customer behavior through data analysis.
- Perform data cleaning and preprocessing.
- Handle categorical and numerical features.
- Build a Machine Learning classification model.
- Predict whether a customer will churn or not.
- Evaluate model performance using different evaluation metrics.
- Save the trained model and preprocessing objects for future use.

## Dataset Information

**Dataset:** Telco Customer Churn Dataset

**Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Dataset Details:

- Total Records: 7,043
- Total Features: 21
- Target Variable: `Churn`
- Problem Type: Binary Classification

### Important Features

| Feature | Description |
|---|---|
| gender | Customer gender |
| SeniorCitizen | Whether customer is a senior citizen |
| Partner | Whether customer has a partner |
| Dependents | Whether customer has dependents |
| tenure | Number of months with the company |
| InternetService | Type of internet service |
| Contract | Type of customer contract |
| MonthlyCharges | Monthly amount charged |
| TotalCharges | Total amount charged |
| Churn | Whether the customer left the service |

## Technologies and Libraries Used

- **Python** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical operations
- **Scikit-learn** - Machine Learning and preprocessing
- **KaggleHub** - Dataset downloading
- **Pickle** - Saving trained model and preprocessing objects
- **PyCharm** - Development environment

## Machine Learning Algorithm

### Logistic Regression

Logistic Regression is a supervised Machine Learning algorithm used for classification problems.

In this project, Logistic Regression is used to classify customers into two categories:

- **0 - No Churn:** Customer continues using the service.
- **1 - Churn:** Customer leaves the service.

## Project Workflow

1. **Dataset Collection:** Downloaded the Telco Customer Churn dataset using KaggleHub.
2. **Data Exploration:** Examined dataset shape, columns, data types, missing values, and duplicate records.
3. **Data Cleaning:** Handled blank values in the TotalCharges column.
4. **Data Preprocessing:** Converted categorical features into numerical values using Label Encoding.
5. **Feature Scaling:** Applied StandardScaler to scale numerical features.
6. **Data Splitting:** Divided the dataset into training and testing sets.
7. **Model Training:** Trained a Logistic Regression classification model.
8. **Model Evaluation:** Evaluated the model using accuracy, precision, recall, F1-score, classification report, and confusion matrix.
9. **Model Saving:** Saved the trained model, scaler, and encoders using Pickle.

## Model Performance and Results

The Logistic Regression model was evaluated on 1,409 test records.

### Accuracy

**77.71%**

### Classification Report

| Metric | No Churn (0) | Churn (1) |
|---|---:|---:|
| Precision | 0.80 | 0.63 |
| Recall | 0.92 | 0.38 |
| F1-Score | 0.86 | 0.47 |
| Support | 1036 | 373 |

### Confusion Matrix

```text
[[955  81]
 [233 140]]
```

### Confusion Matrix Interpretation

- True Negative (TN): 955
- False Positive (FP): 81
- False Negative (FN): 233
- True Positive (TP): 140

### Key Observations

- The model achieved an overall accuracy of 77.71%.
- It performs better at identifying customers who do not churn.
- It correctly identifies 140 customers who are likely to leave.
- The recall for churn customers is 38%, indicating scope for improvement in identifying customers at risk of leaving.

## Project Structure

```text
Customer-Churn-Prediction/
│
├── main.py
├── app.py
├── model.pkl
├── encoders.pkl
├── scaler.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

### File Description

| File | Description |
|---|---|
| main.py | Main Machine Learning implementation |
| app.py | Application code |
| model.pkl | Saved Logistic Regression model |
| encoders.pkl | Saved Label Encoders |
| scaler.pkl | Saved StandardScaler |
| requirements.txt | Required Python libraries |
| .gitignore | Specifies files excluded from Git |

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/krapalidwivedi-tech/Customer-Churn-Prediction.git
```

### 2. Navigate to Project Directory

```bash
cd Customer-Churn-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Machine Learning Script

```bash
python main.py
```

The dataset is downloaded using KaggleHub, and the script trains and evaluates the Logistic Regression model.

## Future Improvements

- Improve churn prediction recall.
- Handle class imbalance using suitable techniques.
- Experiment with Random Forest, Decision Tree, and other classification algorithms.
- Perform hyperparameter tuning.
- Improve model performance through feature engineering.
- Enhance the application interface.

## Conclusion

This project demonstrates the practical implementation of a Machine Learning classification model for predicting customer churn.

It covers the complete Machine Learning workflow, including dataset collection, data cleaning, preprocessing, model training, evaluation, and model saving.

The project helped develop practical understanding of supervised learning, data preprocessing, classification algorithms, and model evaluation.

## Author

**Krapali Dwivedi**

MCA (AI & IoT)

GitHub: [krapalidwivedi-tech](https://github.com/krapalidwivedi-tech)



⭐ If you find this project useful, feel free to explore the repository!
