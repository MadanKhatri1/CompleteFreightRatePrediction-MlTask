<h3 align="center">Predicting Freight Rates</h3>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/pulls)

</div>

<p align="center">
  A machine learning project that trains an ensemble regression model and generates predicted freight rates for unseen shipping loads. Achieved an RMSE of 526.4839, representing a 5.73% reduction in error compared to the baseline model.
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

The final model is built in two stages. First, multiple base models are trained on the target-encoded features:

- **XGBoost**
- **LightGBM**
- **Linear Regression**
- **RealMLP**

Out-of-fold predictions are generated for each base model. These predictions are then passed to a RidgeCV meta-model, which learns how to combine the individual model outputs into one final freight-rate prediction, achieving an RMSE of 526.4839—a 5.73% reduction in error compared to the baseline model's RMSE of 558.5001.

These were the meta model weights 
<img width="760" height="238" alt="image" src="https://github.com/user-attachments/assets/c69a0053-2546-4142-94dd-788456679e62" />

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

### 1. Create a Virtual Environment

Using Python:

```bash
python -m venv Projeenv
```

Using Conda:

```bash
conda create --name Projeenv python=3.12
```

### 2. Activate the Virtual Environment

Using Python:

```bash
Projeenv\Scripts\activate
```

Using Conda:

```bash
conda activate Projeenv
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

1. Run all cells in `modeltraining.ipynb` to train the models and create `stacking_ensemble_pipeline.pkl`.
2. Run all cells in `generate_prediction.ipynb` to generate predictions.
3. Validate the generated files using:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```

## Usage <a name="usage"></a>

The generated prediction files are:

- `validation_predictions.csv` for the validation dataset
- `data/december-chart-inputs.csv` for the December 2025 predictions

The scoring script validates the prediction format and creates a December prediction chart in the `scorer_results` directory.

## Technology Stack <a name="tech_stack"></a>

- Python 3.12
- Pandas and NumPy for data processing
- Scikit-learn for preprocessing and model evaluation
- Linear Regression for modeling the baseline linear relationship and base model
- RidgeCV for combining the base-model predictions in the stacking ensemble
- XGBoost and LightGBM for gradient-boosting models
- RealMLP for neural-network-based tabular regression
- Joblib for saving and loading the trained pipeline
- Matplotlib for prediction visualization

## Contributing <a name="contributing"></a>

Contributions are welcome. Create a new branch, make your changes, and open a pull request with a clear description of the improvement.

## Authors <a name="authors"></a>

- [@MadanKhatri1](https://github.com/MadanKhatri1) 


## Acknowledgments <a name="acknowledgments"></a>

Thanks to the open-source Python and machine-learning communities for the libraries used in this project.

Special thanks to [Mahog](https://www.kaggle.com/mahoganybuttstrings) for the helpful Kaggle solution writeup and notebook. The feature-engineering, target-encoding, and ensemble-modeling ideas from the writeup were valuable references while developing this freight-rate prediction project.

You can find the original Kaggle solution here:

- [1st Place solution](https://www.kaggle.com/competitions/playground-series-s6e1/writeups/1st-place-ive-ran-out-of-catchy-phrases-v)
