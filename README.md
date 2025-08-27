# NN_WindLoadEstimation

## Overview

This repository provides a framework for estimating wind loads using different machine learning models. Feature engineering, training and evaluation of the models in different datasets is done. 

## Data Description

- **Source:** All data comes from wind load simulations.
- **Data Split:** For each file in `data/raw/` (e.g., `wind_speed_11_n.csv`, `wind_speed_13_n.csv`, etc.), 60% of the data is used for training, 20% for testing, and 20% for validation.
- **Wind Speed:** The number in each filename (e.g., `11` in `wind_speed_11_n.csv`) indicates the average wind velocity in the x direction for that simulation.

## Workflow

- **Feature Engineering:** 
  - `nn_est/features.py` and `nn_est/features_new_test_data.py` are used to extract, scale, and create time series sequences from the raw data for use by the different models.
- **Hyperparameters:** 
  - All model and training hyperparameters are defined in `nn_est/hyperparameters.py`.
- **Training:** 
  - `nn_est/train.py` is used to train all models on the wind load simulation data inside `data/raw/`
- **Evaluation:** 
  - `nn_est/evaluate.py` evaluates trained models on the same data used for training/testing/validation.
  - `nn_est/evaluate_new_test_data.py` evaluates models on the friction change datasets inside `data/raw/Systol Files/Fc`
## Models

- **Location:** `models/`
- **Architectures:** Includes FFNN, CNN, LSTM, CNN-LSTM, TCN, Transformer, Mamba, RBF and One Layer Neural Network. Each model is implemented in its own file.

## Results & Logs
- **Plots:** Plots are stored in `plots/`.
- **Logs:** Training and evaluation logs are stored in `logs/`.

## Getting Started

1. **Install dependencies** (`requirements.txt`).
2. **Place simulation data** in `data/raw/`.
3. **Train a model** with `nn_est/train.py`.  
   *Inside `train.py`, select the model you want to train by calling its corresponding training function (e.g., `train_ffnn`, `train_transformer`, etc.) at the end of the file*
4. **Evaluate** with `nn_est/evaluate.py` or `nn_est/evaluate_new_test_data.py`.  
   *In `evaluate.py`, choose the model to evaluate by calling its evaluation function (e.g., `evaluate_ffnn`, `evaluate_transformer`, etc.) at the end of the file.
   In `evaluate_new_test_data.py`, select the model and use the evaluation function at the end of the file.
5. **Check results** in `plots/` and `test_logs/`
