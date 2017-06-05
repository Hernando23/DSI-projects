### Project 4: Disaster Relief + Classification

#### Introduction

The project is based on the famous Titanic dataset. I have conducted a study of Titanic disaster to prepare and  test a classification model that would be useful in disaster analysis and relief in hypothetical research firm that specializes in emergency management.

The study's goal was to predict whether passenger of Titanic survived or not, using various classification techniques.

#### Methodology

Firstly, I carried out an explanatory data analysis, wherein I have dealt with incomplete data, engineered new categories and explored interesting relationships between features by way of data visualtisation. This stage of the project helped me to understand which features are likely to be good predictors for survivorship. For example, features such as age and gender appeared to have been associateed with survival ratio.
Next stage of the study was to deploy various classification models and assess their performance using appropriate metrics such as accuracy, precision, recall and AUC. Models trained and tested in the process include: Logistic Regression, Logistic Regression with Ridge regularization, kNN and Decision Trees.

#### Findings

The table below shows accuracy scores for our models:

Model |	Accuracy score
--- | ---
Logistic Regression |	0.79 (0.77 CV)
Ridge |	0.78 (0.78 CV)
kNN |	0.76
Logistic Regression with avg precision scoring | 0.79
Decision Trees | 0.78
Bagging using Decision Trees | 0.82

Based on the above it apears that the best model to use based on accuracy score is Bagging with Decision Trees which returned the score of 0.82. This model also gives us relatively high precision and recall scores. These metrics are important for the company dealing with disaster relief as they tell us how far can we trust the numbers returned by a particular model.
