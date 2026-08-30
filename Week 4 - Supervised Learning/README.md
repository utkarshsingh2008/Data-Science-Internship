# Week 4 – Supervised Learning

## Breast Cancer Classification

This project is part of my Data Science Internship Week 4 task. In this project, I built a supervised machine learning model to classify breast cancer observations as **Malignant** or **Benign**.

### What I Did

- Loaded and explored the dataset
- Checked missing values and duplicate records
- Prepared features and target variable
- Split the data into training and testing sets
- Applied feature scaling using StandardScaler
- Trained a Logistic Regression model
- Used 5-Fold Cross-Validation
- Evaluated the model using Accuracy, Precision, Recall and F1-Score
- Created a confusion matrix
- Performed detailed error analysis
- Analyzed Logistic Regression coefficients
- Compared Logistic Regression with Random Forest

### Results

Logistic Regression achieved:

- **Test Accuracy:** 98.25%
- **Precision:** 98.61%
- **Recall:** 98.61%
- **F1-Score:** 98.61%
- **Cross-Validation Accuracy:** 98.02%
- **Correct Predictions:** 112 out of 114

I also compared the model with Random Forest. Logistic Regression performed better on the selected test set.

### Tools & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

### Files

- `prog.py` – Python implementation
- `README.md` – Project overview and results
  
### Note

This project is created for learning and internship purposes. The model should not be considered a medical diagnosis tool.
