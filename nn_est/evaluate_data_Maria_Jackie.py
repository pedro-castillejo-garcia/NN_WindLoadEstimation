# Added the original code we used in February
# Needs to be adapted to our current structure

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from pytorch_tcn import TCN

import re
from google.colab import files

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


file_paths_test = [
    'https://raw.githubusercontent.com/moraru-maria/SC-Deep-Learning/refs/heads/main/Data_Test/Step/FcBv_80_80_new.csv',
    'https://raw.githubusercontent.com/moraru-maria/SC-Deep-Learning/refs/heads/main/Data_Test/Step/FcBv_80_80_old.csv',
    'https://raw.githubusercontent.com/moraru-maria/SC-Deep-Learning/refs/heads/main/Data_Test/Step/Fc_80_new.csv',
    'https://raw.githubusercontent.com/moraru-maria/SC-Deep-Learning/refs/heads/main/Data_Test/Step/Fc_80_old.csv',
    'https://raw.githubusercontent.com/moraru-maria/SC-Deep-Learning/refs/heads/main/Data_Test/Var/FcBv_var_new.csv',
    'https://raw.githubusercontent.com/moraru-maria/SC-Deep-Learning/refs/heads/main/Data_Test/Var/FcBv_var_old.csv',
    'https://raw.githubusercontent.com/moraru-maria/SC-Deep-Learning/refs/heads/main/Data_Test/Var/Fc_var_new.csv',
    'https://raw.githubusercontent.com/moraru-maria/SC-Deep-Learning/refs/heads/main/Data_Test/Var/Fc_var_old.csv'
]

# Model in evaluation mode
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

for file in file_paths_test:
    df_test = pd.read_csv(file)
    # testing
    x_test=df_test[features].values
    y_test=df_test[targets].values
    # Normalize test features and targets
    x_test = scaler.transform(x_test)
    y_test = scaler_y.transform(y_test)
    # Create PyTorch dataset
    test_dataset = WindTurbineDataset(x_test, y_test)
    # test split
    test_set = test_dataset
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    actual_values = []
    predicted_values = []

    with torch.no_grad():
      for x_batch, y_batch in test_loader:
          x_batch = x_batch.to(device)
          y_batch = y_batch.to(device)
          x_batch = x_batch.unsqueeze(2)
          outputs = model(x_batch)
          actual_values.append(y_batch.cpu().numpy())
          predicted_values.append(outputs.cpu().numpy())

    actual_values = np.concatenate(actual_values, axis=0)
    predicted_values = np.concatenate(predicted_values, axis=0)

    # Invert the scaling to get original values
    # Separate scalers for X and y
    actual_values_inv = scaler_y.inverse_transform(actual_values)
    predicted_values_inv = scaler_y.inverse_transform(predicted_values)

    time_values = df_test['t'].values

    # Create a DataFrame
    results_df = pd.DataFrame({
        'Time': time_values,  # assuming you have a time_values array
        'Actual_Mz1': actual_values_inv[:, 0],
        'Predicted_Mz1': predicted_values_inv[:, 0],
        'Actual_Mz2': actual_values_inv[:, 1],
        'Predicted_Mz2': predicted_values_inv[:, 1],
        'Actual_Mz3': actual_values_inv[:, 2],
        'Predicted_Mz3': predicted_values_inv[:, 2]
    })

    file_name = re.findall(r'[^/]+$', file)[0]
    # Save DataFrame to CSV
    results_df.to_csv(f'{file_name}.csv', index=False)
    files.download(f'{file_name}.csv')



# Calculate the metrics
mae = mean_absolute_error(actual_values_inv, predicted_values_inv)
mse = mean_squared_error(actual_values_inv, predicted_values_inv)
rmse = np.sqrt(mse)
r2 = r2_score(actual_values_inv, predicted_values_inv)

# Print the results
print("Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")