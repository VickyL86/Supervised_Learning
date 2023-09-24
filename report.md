# Module 12 Report Template

## Overview of the Analysis

The purpose of the analysis was to use a supervised machine learning model to evaluate and predict loan risk. The analysis focused on lending data by identifying and distinguishing high-risk loan records. The 'loan_status' column was used as the target - total number of records used: 75,036 healthy loans, and 2500 high-risk loans [total: 77,536].     

Used Libraries:
- Numpy
- Pandas
- Sklearn 

1) Data Preprocessing:

- Split the dataset into features (X) and target (y).
- Further split the data into training and testing sets using train_test_split, ensuring stratified sampling due to class imbalance.

2) Modeling:

- Implemented Logistic Regression model, utilizing the lbfgs solver.
- Fitted the model on the training data.

3) Evaluation:

- Made predictions using the test data.
- Evaluated the model using confusion matrix and classification report to assess accuracy, precision, recall, and F1-score.


## Results

Machine Learning Model 1: Logistic Regression

Accuracy: Achieved an overall accuracy of 99%, indicating a high proportion of correct predictions.

Precision: Demonstrated a precision of 87% for "High-risk" loans, meaning 87% of loans predicted as "High-risk" were correctly identified, while 13% were falsely labeled.

Recall: Achieved a recall rate of 89% for "High-risk" loans, indicating that the model correctly identified 89% of all actual "High-risk" loans, but missed 11%.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
4025