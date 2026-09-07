# Week 6 – Integrative Capstone Project

## Customer Churn Prediction and Customer Segmentation

This project is part of my Data Science Internship Week 6 task. In this project, I combined the concepts learned throughout the internship into one complete data science workflow.

I used the Telco Customer Churn Dataset to analyze customer behavior, predict customer churn, and identify different customer segments.

## What I Did

- Loaded and explored the dataset
- Checked missing values and duplicate records
- Cleaned and preprocessed the data
- Performed Exploratory Data Analysis (EDA)
- Analyzed customer churn patterns
- Applied feature encoding and scaling
- Built a Logistic Regression model for churn prediction
- Evaluated the model using Accuracy, Precision, Recall and F1-Score
- Created a confusion matrix and performed error analysis
- Applied K-Means clustering for customer segmentation
- Evaluated the clusters using the Silhouette Score
- Built a Neural Network using PyTorch
- Compared Logistic Regression with the Neural Network
- Identified key insights and provided recommendations

## Results

### Logistic Regression
- Accuracy: 80.38%
- Precision: 64.76%
- Recall: 57.49%
- F1-Score: 60.91%

### Neural Network
- Accuracy: 78.68%
- Precision: 62.09%
- Recall: 50.80%
- F1-Score: 55.88%

Logistic Regression performed better than the Neural Network on the selected test dataset. The results also showed that improving Recall is important because some actual churn customers were not identified by the model.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- PyTorch

## Dataset

Telco Customer Churn Dataset

The dataset contains customer information such as tenure, services, contract type, monthly charges, total charges, and churn status.

## Project Structure

Week-6-Integrative-Capstone/
- README.md
- prog.py

## Conclusion

This project helped me combine data preprocessing, EDA, supervised learning, unsupervised learning, and deep learning into one end-to-end project. It also helped me understand how model evaluation and customer segmentation can be used together to analyze customer churn.

Note: This project was created for learning and internship purposes.
