# COMP1827

## Overview

This repository contains scripts for optimising and evaluating machine learning
models for housing price prediction.

## Dataset

The dataset used is the
[California Housing Prices dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices).

## Structure

- **Optimisers**: Scripts prefixed with `optimiser_` are used for hyperparameter
  tuning.
- **Models**: Scripts prefixed with `model_` are used for training and
  evaluating machine learning models.

> **Note**:
>
> All models have already been optimized.

## Results

- The best-performing model is **XGBoost**, with a **Success Score** of 79.18%.
- **Random Forest** is a strong alternative, trailing XGBoost by 1.54%.
- **Linear Regression** performed the worst due to its inability to handle
  non-linear relationships in the dataset.

### Summary

The results align with expectations:

- **XGBoost** generally outperforms Random Forest due to its advanced boosting
  techniques.
- **Linear Regression** is unsuitable for this task because housing prices
  exhibit non-linear relationships.
