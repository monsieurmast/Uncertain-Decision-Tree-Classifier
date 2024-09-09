# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:01:21 2024

@author: sstel
"""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


# =============================================================================
# Main Function: Adapt Dataframe
# =============================================================================
def adapt_df(X, y, w, u, random_seed):
    if u > 0:
        
        X = perturbe_values(X, u, random_seed)
    
    X = normalize_dataframe(X)
    print(X.shape)
    
    return fit_error_model(X, y, w)

# =============================================================================
# Fit error model
# =============================================================================
def fit_error_model(X, y, w):
    
    print(X.shape)
    udf = []
    
    for i, row in X.iterrows():
        
        current_row = []
        
        for j, col in enumerate(row):
                
            mean = col
         
            lower_bound, upper_bound = calculate_bounds(mean, w)
            std = (upper_bound - lower_bound) / 4

            current_row.append(mean)
            current_row.append(std)
            current_row.append(lower_bound)
            current_row.append(upper_bound)
        
        # Append the new row to the existing DataFrame
        udf.append(current_row)
        
    udf = np.array(udf)
    y = np.array(y)
    y = y.astype(np.int32)

    return udf, y

# =============================================================================
# Perturbe Values
# =============================================================================
def perturbe_values(X, u, random_seed):
    X = normalize_dataframe(X)
    
    tuple_size = len(X.index)
    n_columns = len(X.columns)
    
    for i in range(tuple_size):
        
        for j in range(n_columns):
            
            original_mean = X.iloc[i, j]
            
            np.random.seed(random_seed + i * n_columns + j)
            std = u / 4
            
            error = np.random.normal(0, std, size=1)
            X.iloc[i, j] = original_mean + error

    X = normalize_dataframe(X)

    return X

# =============================================================================
# Adapt Data Input for DTC use
# =============================================================================

def normalize_dataframe(X):
    
    # Use MinMaxScaler to normalize each attribute to the range [0, 1]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(X.values)
    normalized_df = pd.DataFrame(normalized_data, columns=X.columns, index=X.index)
    
    return normalized_df

# =============================================================================
# Calculate Bounds for Uncertain Attribute
# =============================================================================

def calculate_bounds(mean, w):
    
    lower_bound = mean - w / 2
    upper_bound = mean + w / 2
    
    return lower_bound, upper_bound

# =============================================================================
# Fetch n random tuples from a dataset
# =============================================================================
def fetch_random_data(dataset, n, random_seed):
    np.random.seed(random_seed)  # Set the seed for reproducibility
    
    # Use the sample method directly on the dataset to get n random rows
    random_subset_indices = np.random.choice(len(dataset.data), n, replace=False)
    
    X = pd.DataFrame(dataset.data[random_subset_indices])
    y = pd.Series(dataset.target[random_subset_indices])
    
    return X, y.astype(int)


