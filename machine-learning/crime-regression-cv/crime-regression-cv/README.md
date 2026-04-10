# Regression and Cross-Validation in R

## Overview
This project applies multiple regression techniques to real-world datasets including crime statistics and exercise health data. The goal is to compare predictive performance using both traditional and robust statistical methods.

## Methods Used
- Multiple Linear Regression
- Robust Regression (Huber and Tukey’s Bisquare)
- Iteratively Reweighted Least Squares (IRLS)
- k-Nearest Neighbors (kNN)
- Cross-Validation (10-fold and LOOCV)
- Model selection using MAE and MSE

## Key Concepts
- Effect of outliers on regression models
- Robust statistical techniques
- Model evaluation using cross-validation
- Bias-variance tradeoff in kNN vs linear models

## Tools
- R
- tidyverse
- ggformula
- caret
- MASS
- FNN
- ISLR

## Key Insight
Robust regression methods improve model stability in the presence of outliers, and cross-validation provides an unbiased way to compare models.

## Conclusion
This project demonstrates that model choice significantly affects predictive accuracy, especially when outliers are present.
