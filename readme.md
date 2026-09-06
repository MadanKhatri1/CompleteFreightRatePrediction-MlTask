<h3 align="center">Predicting Freight Rates</h3>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/pulls)

</div>

<p align="center">
  A machine learning project that trains an ensemble regression model and generates predicted freight rates for unseen shipping loads.
</p>

## 📝 Table of Contents
- [Problem Statement](#problem_statement)
- [Idea / Solution](#idea)
- [Installation](#Installation)
- [Usage](#usage)
- [Technology Stack](#tech_stack)
- [Contributing](./CONTRIBUTING.md)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## 🧐 Problem Statement <a name="problem_statement"></a>
The goal of this project is to predict the posted rate for freight loads.

Each load contains information about:

- Pickup and delivery locations
- Pickup and delivery coordinates
- Distance
- Equipment type
- Weight
- Date
- Market index
- Quote signal

The target variable is `posted_rate`, which represents the expected price for transporting the load.

## Solution <a name="idea"></a>

The solution combines **feature engineering**, **target encoding**, and a **stacking ensemble**.

### 1. Feature Engineering

The following features are created:

- **Hour**, **month**, and **day of the week** extracted from the load date
- **Sine and cosine date encodings** to represent cyclical time patterns
- **Squared weight** and **squared distance**
- Duplicate numeric features for model compatibility
- A manually designed **linear pricing formula**
- A combined categorical feature called `equipment_delivery_hub`

Missing values in `weight` and `market_index` are filled using averages calculated from the training data.

### 2. Target Encoding

The following categorical columns are target encoded:

- `pickup`
- `delivery`
- `equipment`
- `equipment_delivery_hub`

Target encoding replaces each category with statistical information calculated from the target variable, `posted_rate`.

**Empirical-Bayes smoothing** is used for mean encoding. This reduces the effect of categories with very few observations and produces more stable estimates.

During training, cross-validation is used so that each validation fold is encoded without using its own target values. This helps prevent target leakage.

During prediction, the fitted encoder stored inside the pipeline is used to transform new data.

### 3. Stacking Ensemble

The final model combines predictions from multiple base models:

- **XGBoost**
- **LightGBM**
- **Linear Regression**
- **RealMLP**

The predictions from these models are passed to a **RidgeCV meta-model**. The meta-model learns how to combine the individual predictions into one final freight-rate prediction.

The complete ensemble is saved in:

```text
stacking_ensemble_pipeline.pkl
```
This file contains
- The fitted target encoder
- The fitted base models
- The fitted meta-model

### 4. Prediction Workflow

The prediction process follows these steps:

1. Load the saved stacking pipeline.
2. Load the training reference, validation, and December datasets.
3. Recreate the same engineered features used during training.
4. Apply the fitted target encoder from the saved pipeline.
5. Generate predictions using each base model.
6. Combine the base-model predictions using the RidgeCV meta-model.
7. Replace negative predictions with a small positive value.
8. Save the final prediction files.

## Installation <a name="Installation"></a>
