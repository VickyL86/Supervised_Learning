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
- Used the standard scaler before fitting the random forest model.

2) Modeling:

- Implemented Logistic Regression model, utilizing the lbfgs solver.
- Implement Random Forest model, utilizing RandomForestClassifier.
- Fitted the model on the training data - applied .ravel() to flatten the data when fitting the random forest model. 

3) Evaluation:

- Made predictions using the test data.
- Evaluated the model using confusion matrix and classification report to assess accuracy, precision, recall, and F1-score.
- Included the false positive/false negative matrix. 
- added the feature importance breakdown to the random forest analysis.


## Results

* Machine Learning Model 1: Logistic Regression

Accuracy: Achieved an overall accuracy of 99.24%, indicating a high proportion of correct predictions.

Precision: Demonstrated a precision of 87% for "High-risk" loans, meaning 87% of loans predicted as "High-risk" were correctly identified, while 13% were falsely labeled.

Recall: Achieved a recall rate of 89% for "High-risk" loans, indicating that the model correctly identified 89% of all actual "High-risk" loans, but missed 11%.


* Machine Learning Model 2: Radom Forest 

Accuracy: Achieved an overall accuracy of 99.15%, indicating a high proportion of correct predictions.

Precision: Demonstrated a precision of 85% for "High-risk" loans, meaning 85% of loans predicted as "High-risk" were correctly identified, while 15% were falsely labeled.

Recall: Achieved a recall rate of 88% for "High-risk" loans, indicating that the model correctly identified 88% of all actual "High-risk" loans, but missed 12%.

## Summary

The random forest forest model exhibits strong performance when distinguishing between health and high-risk loans showing the overall accuracy score of 99.16%. The accuracy score achieved by the random forest model is only very slightly lower than that achieved with the logitisc regression model (99.24%). This means that both models are are highly accurate in predicting the loan status.


PRECISION: 
For "High-risk loans", the Random Forest model has a precision of 85%, slightly lower than the Logistic Regression model (87%). This means the Random Forest model has a slightly higher rate of falsely labeling healthy loans as high-risk.

RECALL:

The recall for high-risk loans in the Random Forest model is 88%, which is very similar to the recall of 89% in the Logistic Regression model. This suggests that both models are almost equally good at identifying the actual high-risk loans.


F1-SCORE:

The F1-score for high-risk loans in the Random Forest model is 87%, slightly lower than the 88% in the Logistic Regression model. The F1-score gives a balanced measure of precision and recall, and in this case, the difference is marginal.

FALSE POSITIVES/FALSE NEGATIVES:

The Random Forest model resulted in 93 False Positives and 70 False Negatives, compared to the Logistic Regression model of 80 and 67 respectively. For a more risk-averse bank with enough applicants not to fear losing customers, logistic regression model might be a better choice since its confusion matrix indicates a slightly lower Type 2 error. 

Scaling and Complexity:

The Random Forest model required additional steps like scaling the features, and it’s generally more complex and computationally intensive compared to Logistic Regression. Depending on the available computational resources and the need for interpretability, Logistic Regression might be preferred.

Conclusion:

Both models exhibit strong and comparable performance in classifying loan status. The feature importance analysis for the Random Forest model reveals that 'interest_rate', 'borrower_income', 'debt_to_income', 'total_debt', and 'loan_size' are the most strongest predictors of loan status. This helps to understand which features are driving the model's predictions and can inform risk management strategies and policy development for the bank.

The choice between Logistic Regression and Random Forest may depend on specific business requirements, such as the importance of minimizing false positives/negatives, computational efficiency, model interpretability, and gaining insights into feature importance. Given the minor differences in performance metrics, if computational efficiency, model simplicity, and interpretability are priorities, Logistic Regression might be a more suitable choice.

However, if the model's ability to generalize well, handle non-linear relationships, and provide insights into the relative importance of different features is more critical, the Random Forest model may have the edge. The detailed feature importance provided by the Random Forest model can be particularly useful for refining and optimizing the loan approval process and for identifying areas where additional data collection or feature engineering may improve model performance.

In conclusion, the decision to use either model should be aligned with the specific goals and constraints of the financial institution, with consideration given to the trade-offs between interpretability, complexity, accuracy, and the value of understanding feature influence on predictions.
