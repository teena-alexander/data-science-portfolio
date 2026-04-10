# Cross-Validation for Model Selection (R)

## Overview
This project explores predictive modeling using linear regression and k-Nearest Neighbors (kNN) to predict systolic blood pressure based on exercise and demographic variables.

The main focus is on model evaluation using cross-validation (LOOCV and 10-fold CV).

## Methods Used
- Data cleaning and outlier removal
- Linear regression modeling
- k-Nearest Neighbors regression
- One-hot encoding for categorical variables
- Cross-validation (10-fold)
- Model comparison using MAE and MSE

## Tools
- R
- tidyverse
- ggformula
- FNN
- caret (conceptually for CV)

## Key Insight
Cross-validation was used to fairly compare models and select the best-performing model based on MAE.

## Files
- `BP_and_exercise.Rmd` → full analysis code
- `BP_and_exercise.csv` → dataset
- `output.docx` → knitted report (optional)

## Results Summary
Best model was selected based on lowest cross-validated MAE.
