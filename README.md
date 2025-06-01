# Creditworthiness Prediction System

## 📌 Project Description

This project predicts an individual's creditworthiness (good or bad credit risk) using historical financial and personal data. It leverages machine learning classification models to analyze key indicators like credit history, loan amount, duration, job type, and more.

The dataset used is the **Statlog (German Credit Data)** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)).

---

## ✅ Features

- Load and preprocess real-world financial data
- Encode categorical features
- Normalize numerical values using `StandardScaler`
- Train classification models (Logistic Regression, Random Forest)
- Evaluate with metrics: **Precision, Recall, F1-Score, ROC-AUC**
- Save trained model for future use
- Predict creditworthiness for new users via a reusable function

---

## 🛠️ Tools and Technologies Used

- **Python 3**
- **Pandas** – data handling
- **NumPy** – numerical operations
- **Scikit-learn** – ML models, preprocessing, evaluation
- **Matplotlib / Seaborn** – data visualization
- **Joblib** – model persistence
- **ucimlrepo** – for fetching UCI datasets

---

## 📂 Project Structure

```plaintext
.
├── Task1_codealpha.ipynb     # Main notebook with all code
├── credit_model.pkl          # Trained model (to be generated)
├── scaler.pkl                # Saved StandardScaler object
├── model_columns.pkl         # Saved column order for prediction
├── README.md                 # Project documentation (this file)
