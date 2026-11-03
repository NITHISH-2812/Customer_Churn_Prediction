# Customer Churn Prediction

## 1. Project Overview

Customer Churn Prediction is a machine learning project that predicts whether a customer is likely to leave a company's service.

The project uses customer information such as tenure, contract type, monthly charges, internet service, and payment method to predict customer churn.

A Streamlit web application allows users to enter customer details and receive a churn prediction with an estimated probability.

## 2. Objectives

- Analyze customer data to identify churn patterns.
- Clean and preprocess customer data.
- Train machine learning classification models.
- Evaluate model performance using different metrics.
- Build an interactive web application for churn prediction.

## 3. Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- VS Code

## 4. Dataset

Dataset: IBM Telco Customer Churn

Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains customer information, service details, billing information, and churn status.

## 5. Machine Learning Models

Two classification algorithms were trained:

1. Logistic Regression
2. Random Forest Classifier

The models were evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.

Logistic Regression was selected as the final model based on its F1-score.

## 6. Model Performance

| Metric | Logistic Regression |
|---|---:|
| Accuracy | 80.38% |
| Precision | 64.85% |
| Recall | 57.22% |
| F1-Score | 60.80% |
| ROC-AUC | 83.59% |

## 7. Application Features

- Interactive customer input form.
- Customer churn prediction.
- Estimated churn probability.
- Model performance dashboard.
- Confusion matrix display.

## 8. Project Structure

```text
Customer_Churn_Prediction/
│
├── dataset/
│   └── customer_churn.csv
│
├── model/
│   └── churn_model.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## 9. How to Run the Project

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the machine learning model

```bash
python train_model.py
```

### Step 3: Run the Streamlit application

```bash
python -m streamlit run app.py
```

### Step 4: Open the application

The application will open in your web browser. Enter customer details and click the Predict Churn button.

## 10. Sample Testing

Two sample customer profiles were tested:

- Customer 1: 0.6% estimated churn probability — likely to stay.
- Customer 2: 79.5% estimated churn probability — likely to churn.

These are sample application tests, not proof of overall model accuracy.

## 11. Conclusion

This project demonstrates how machine learning can be used to predict customer churn.

Logistic Regression achieved an accuracy of 80.38% and an ROC-AUC score of 83.59% on the test dataset.

The Streamlit application provides a simple interface for making predictions and understanding the model's performance.

The project can be extended in the future by improving churn detection, tuning prediction thresholds, and adding more visualizations.
