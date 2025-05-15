# Credit Risk Scoring

This project implements a **machine learning pipeline** for credit risk scoring using logistic regression. The pipeline includes data preprocessing, model training, evaluation, and feature importance visualization.

---

## Project Overview

The goal is to predict the likelihood of a loan default based on borrower data. The project involves:

- Data loading and preprocessing (encoding categorical variables, scaling features)
- Splitting data into training and test sets
- Training a logistic regression model with missing value imputation
- Evaluating model performance with accuracy and classification reports
- Visualizing feature importance using logistic regression coefficients

---

## Repository Structure

![image](https://github.com/user-attachments/assets/f005736c-2486-4d13-9312-7979c1c9f3b6)

---

## Dataset

This project uses the [Credit Risk Dataset from Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset), which contains loan application data used for credit risk scoring.

---

## How to Use

1. **Data Preprocessing and Splitting**  
Run `01_data_preprocessing_and_split.ipynb` to load the dataset, encode categorical variables, scale features, and split into training and testing sets. This saves the processed data and scaler objects.

2. **Model Training with Imputation**  
Run `02_model_training_with_imputation.ipynb` to load processed data, apply missing value imputation, train a logistic regression model, and save the trained model and predictions.

3. **Model Evaluation**  
Run `03_model_evaluation.ipynb` to evaluate the model's performance on the test set using accuracy and classification report metrics.

4. **Feature Importance Visualization**  
Run `04_feature_importance_visualization.ipynb` to visualize the logistic regression coefficients indicating the importance of each feature.

---

## Dependencies

- Python
- pandas
- scikit-learn
- joblib
- matplotlib
- numpy

You can install required packages with:

```bash
pip install -r requirements.txt



