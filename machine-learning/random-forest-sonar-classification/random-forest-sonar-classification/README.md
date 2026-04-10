# Sonar Signal Classification Using Random Forest

## Overview
This project builds a machine learning model to classify sonar signals as either rock or metal using a Random Forest classifier.

The dataset contains 60 numerical features representing sonar frequency readings.

## Objective
Predict whether an object is a rock or metal based on sonar return signals.

## Methods Used
- Random Forest classification
- Hyperparameter tuning (mtry selection)
- 5-fold cross-validation
- Feature importance analysis
- Partial dependence plots
- Double cross-validation for unbiased accuracy estimation

## Tools
- R
- caret
- randomForest
- dplyr
- ggformula

## Key Insights
- Random Forest performs significantly better than the no-information baseline (~53.4%).
- Feature importance shows which sonar frequencies contribute most to classification.
- Partial dependence plots help interpret model behavior.

## Evaluation
- Cross-validation accuracy used for model selection
- Confusion matrix used for final performance evaluation

## Key Output
Final model demonstrates strong predictive ability for distinguishing sonar signals.

