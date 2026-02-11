# Hrishikesh_ML_Assignment_02
Hrishikesh_Streamlit_app_for_ML_classification_models_assignment

Problem Statement:

Financial institutions face significant risk due to credit card payment defaults. Accurately predicting whether a customer will default on their credit card payment in the next month is critical for minimizing financial losses and improving risk management strategies.
The objective of this project is to design and develop a machine learning-based classification system that predicts the likelihood of credit card default using customer demographic, financial, and repayment history data. This is to be done by implementing six classification models Logistic_regression, xgboost_model, random_forest_classifier, naive_bayes, knn, and decision_tree. The metrices to be evaluated should be Accuracy, Precision, Recall, F1 Score, AUC Score, Matthews Correlation Coefficient (MCC).

Dataset Description:

a. Training Dataset:
This project utilizes the UCI Default of Credit Card Clients Dataset, a publicly available dataset from the UCI Machine Learning Repository. The dataset contains financial and demographic information of credit card clients in Taiwan and is commonly used for binary classification tasks in credit risk modeling.

Dataset Characteristics:
Number of instances: 30,000
Number of input features: 23
Target variable: 1 (binary classification)
Task type: Supervised binary classification
The dataset includes both categorical and numerical variables that capture customer demographics, credit exposure, repayment behavior, billing statements, and historical payment records.

Feature Description:
The independent variables are categorized as follows:
1. Demographic Attributes
LIMIT_BAL: Credit limit (NT dollar)
SEX: Gender (1 = male, 2 = female)
EDUCATION: Education level (ordinal categorical)
MARRIAGE: Marital status (categorical)
AGE: Age in years

3. Repayment Status (April–September 2005)
PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6
Repayment status in past six months, measured as ordinal categorical variables representing payment delay status.

4. Bill Statement Amounts
BILL_AMT1 – BILL_AMT6
Monthly bill statement amounts (continuous numerical variables).

4. Previous Payment Amounts
PAY_AMT1 – PAY_AMT6
Amount of previous payments (continuous numerical variables).

Target Variable:
default payment next month (renamed as target)
0 → No default
1 → Default
This binary dependent variable indicates whether the client defaulted on the next month’s payment obligation.

Statistical Nature of the Dataset:
Mixed data types (categorical and continuous)
Moderate class imbalance (non-default cases dominate)
Suitable for evaluation using classification performance metrics such as Accuracy, Precision, Recall, F1-score, AUC-Score, and MCC
Appropriate for testing both linear and non-linear classification models


b. Test Dataset:
The test dataset used in this project is a subset of the UCI Default of Credit Card Clients dataset, created specifically for model evaluation within the deployed Streamlit application.

Dataset Characteristics:
Number of instances: 100
Number of input features: 23
Target variable: 1 (binary classification)
Task type: Supervised binary classification
The test dataset maintains the same structure, feature names, and data types as the original training dataset to ensure consistency in preprocessing and model evaluation.

Feature Structure:
The test dataset includes the same 23 predictor variables categorized as:
1. Demographic Features
LIMIT_BAL
SEX
EDUCATION
MARRIAGE
AGE

2. Repayment Status Variables:
PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6

3. Bill Statement Amounts:
BILL_AMT1 – BILL_AMT6

4. Previous Payment Amounts
PAY_AMT1 – PAY_AMT6

Target Variable:
target
0 → No default
1 → Default
This variable represents whether the customer defaulted on the next month’s credit card payment.

Purpose of the Test Dataset:

The test dataset is used to:
Evaluate trained classification models
Compute performance metrics including Accuracy, Precision, Recall, F1 Score, AUC-ROC, and MCC
Generate confusion matrices
Validate model generalization performance
Since the test dataset preserves the same feature space and preprocessing pipeline as the training dataset, it ensures reliable and consistent evaluation of model performance.

## Models used and their Performance Comparison

| **ML Model Name**          | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1** | **MCC** |
|----------------------------|--------------|---------|---------------|------------|--------|--------|
| Logistic Regression        |   0.7500     |  0.6637 |    0.6000     |   0.1154   | 0.1935 | 0.1778 |
| Decision Tree              |   0.9800     |  0.9615 |    1.0000     |   0.9231   | 0.9600 | 0.9480 |
| kNN                        |   0.8200     |  0.8612 |    0.8333     |   0.3846   | 0.5263 | 0.4827 |
| Naive Bayes                |   0.7700     |  0.7027 |    0.5714     |   0.4615   | 0.5106 | 0.3661 |
| Random Forest (Ensemble)   |   0.9700     |  0.9875 |    1.0000     |   0.8846   | 0.9388 | 0.9220 |
| XGBoost (Ensemble)         |   0.8200     |  0.9137 |    0.8333     |   0.3846   | 0.5263 | 0.4827 |


**OBSERVATIONS ON PERFORMANCE:**

| ML Model Name | Observation about Model Performance                         |
|---------------|------------------------------------------------------------ |
| Logistic Regression | Achieved moderate accuracy (0.75) but very low recall | 
|                     | (0.1154) and F1-score (0.1935), indicating poor       |
|                     | detection of the positive (default) class. The model  |
|                     | struggles with class imbalance and underfits complex  |
|                     | patterns.                                             |
-------------------------------------------------------------------------------
| Decision Tree       | Delivered excellent overall performance with very     |
|                     | high accuracy (0.98), AUC (0.9615), recall (0.9231),  |
|                     | F1-score (0.96), and MCC (0.9480). Demonstrates       |
|                     | strong capability in capturing  nonlinear decision    | 
|                     | boundaries.                                           |
-------------------------------------------------------------------------------
| kNN                 | Showed moderate performance with good accuracy (0.82) |
|                     | but relatively low recall (0.3846). The model detects |
|                     | some positive cases but lacks strong discriminative   |
|                     | power compared to tree-based models.                  |
-------------------------------------------------------------------------------
| Naive Bayes         | Produced balanced but moderate metrics                |
|                     | (Accuracy: 0.77, F1: 0.5106). Assumption of feature   |
|                     | independence may limit performance on correlated      |
|                     | financial attributes.                                 |
-------------------------------------------------------------------------------
| Random Forest       |  Achieved very strong performance across all metrics  | 
|(Ensemble)           |  (Accuracy: 0.97, AUC: 0.9875, MCC: 0.9220). Shows    |
|                     |  high robustness and generalization due to ensemble   |
|                     |  averaging.                                           |
-------------------------------------------------------------------------------
| XGBoost (Ensemble)  | Demonstrated high AUC (0.9137) and good overall       |
|                     | performance but lower recall (0.3846) compared to     |
|                     | Random Forest and Decision Tree, indicating room for  |
|                     | hyperparameter tuning.                                |
-------------------------------------------------------------------------------

