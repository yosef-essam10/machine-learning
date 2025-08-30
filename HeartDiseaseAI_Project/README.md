Heart Disease Prediction

This project is a Machine Learning model that predicts whether a person is likely to have heart disease based on medical attributes from the UCI Heart Disease Dataset.

Dataset

The dataset contains patient health information such as:

Age

Sex

Chest pain type (cp)

Resting blood pressure (trestbps)

Cholesterol (chol)

Fasting blood sugar (fbs)

Resting ECG results (restecg)

Maximum heart rate achieved (thalach)

Exercise induced angina (exang)

Oldpeak (ST depression)

Slope of ST segment (slope)

Number of major vessels (ca)

Thalassemia (thal)

Target variable:

0 → No heart disease

1 → Presence of heart disease

Steps in the Project

Load and clean the dataset

Exploratory Data Analysis (EDA)

Split data into train/test sets

Apply different Machine Learning models:

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Decision Tree

Random Forest

Evaluate models using Accuracy, Precision, Recall, and F1-score

Select the best-performing model

Results

The best-performing model achieved an accuracy of around 85% - 90% on the test set.

Random Forest and Logistic Regression gave the most reliable results.

How to Run

Install dependencies:

pip install -r requirements.txt


Run the Jupyter Notebook or Python script to train and test the models.

Future Work

Deploy the model using Flask/Streamlit

Optimize hyperparameters further

Add deep learning models for comparison