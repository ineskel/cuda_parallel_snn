import os
import numpy as np
import pandas as pd
from enum import Enum

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_samples = int(256*100)       # Number of samples
num_features = 32             # Number of input features
noise_std = 0.1              # Standard deviation of Gaussian noise
input_scale = 1            # Scale of input features

# Generate random input features
X = np.random.uniform(-input_scale, input_scale, size=(num_samples, num_features))

# Define an enum for function choices
class TargetFunction(Enum):
    CONVEX = "convex"

# Define target functions
def convex_target_function(x):
    return np.sum(x**2, axis=1)

# Function to select the target function
def compute_target(function_choice, x):
    if function_choice == TargetFunction.CONVEX:
        return convex_target_function(x)
    else:
        raise ValueError("Invalid function choice")

# Choose the target function to use
function_choice = TargetFunction.CONVEX

# Compute target variable with noise
Y = compute_target(function_choice, X) + np.random.normal(0, noise_std, size=num_samples)

# Create a DataFrame
columns = [f'X{i+1}' for i in range(num_features)] + ['Y']
data = np.hstack((X, Y.reshape(-1, 1)))
df = pd.DataFrame(data, columns=columns)

# Save to CSV
os.makedirs('data', exist_ok=True)
df.to_csv(f'data/synthetic_{function_choice.value}_large.csv', index=False, header=None)
print(f"Dataset 'synthetic_{function_choice.value}.csv' generated successfully.")
