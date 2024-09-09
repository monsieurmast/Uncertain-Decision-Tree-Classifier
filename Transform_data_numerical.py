# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:43:27 2024

@author: Simon Stellmach
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np



from scipy.stats import norm

# =============================================================================
# Main function: transforms dataset for AVG or UDT Usage
# =============================================================================
def transform_dataset(X, y, w, s, random_seed, mean_based, perturbed, u):
    if mean_based:
        return calculate_uncertainty(X, y, w, s, random_seed, mean_based, perturbed, u)
    else: 
        udf, y, split_points, means, stds = calculate_uncertainty(X, y, w, s, random_seed, mean_based, perturbed, u)
    probs = []

    for t, tup in enumerate(udf):
        t_probs = []
        for a, attribute in enumerate(tup):
            lower_bound = attribute[0]
            upper_bound = attribute[s-1]
            split_points_a = np.array(split_points[a])
            t_probs_a = np.zeros(len(split_points_a))

            upper_bound_condition = (upper_bound <= split_points_a)
            lower_bound_condition = (split_points_a < lower_bound)

            nan_condition = np.isnan(split_points_a)

            t_probs_a[upper_bound_condition] = 1
            t_probs_a[lower_bound_condition] = 0
            t_probs_a[nan_condition] = np.nan

            remaining_condition = ~(upper_bound_condition | lower_bound_condition | nan_condition)
            index = np.searchsorted(attribute, split_points_a[remaining_condition])
            
    
            t_probs_a[remaining_condition] = norm.cdf(attribute[index], loc=means[t, a], scale=stds[t, a])

            t_probs.append(t_probs_a)

        probs.append(t_probs)

    return np.array(probs), y

# =============================================================================
# Calculates the probabilities for the samples
# =============================================================================
def calculate_uncertainty(X, y, w, s, random_seed, mean_based, perturbed, u): 

    if perturbed:
        X = perturbe_values(X, u, random_seed)
    
    X = normalize_dataframe(X)
    
    udf = []
    stds = []  
    means = []
    
    for i, row in enumerate(X.values):
        
        current_row = []
        means_t = []
        stds_t = []
        
        for j, col in enumerate(row):
            
            mean = col
            lower_bound, upper_bound = calculate_bounds(mean, w)
            std = (upper_bound - lower_bound) / 4
            samples = calculate_distribution(lower_bound, upper_bound, s)
    
            if mean_based:
                current_row.append(mean)
            else:
                current_row.append(samples)
            
            means_t.append(mean)
            stds_t.append(std)
            
        # Append the new row to the existing DataFrame
        udf.append(current_row)
        means.append(means_t)
        stds.append(stds_t)

    udf = np.array(udf)
    means = np.array(means)
    stds = np.array(stds)
    y = np.array(y).astype(np.int32)
    y = np.squeeze(y)
    
    if mean_based:
        return udf, y
    else:
        sp = calculate_splitting_points(udf, s)
        return udf, y, sp, means, stds


# =============================================================================
# Calculate the unique splitting points for each Attribute
# =============================================================================
def calculate_splitting_points(X, s):
    split_points = []

    # Get the number of features in the array
    num_features = X.shape[1]
    
    for feature in range(0, num_features):        
        sp = []
        for t in range(len(X)):
                samples = X[t, feature, :]
                for sample in samples:
                    sp.append(sample)
        
        sp = np.sort(sp)
        sp = np.unique(sp)
       
        split_points.append(sp)
    
    split_points = pd.DataFrame(split_points)
    split_points = np.array(split_points)
    return split_points
        

# =============================================================================
# Calculate samples within bounds
# =============================================================================
def calculate_distribution(lb, ub, s):
    current = lb
    iterator = (ub - lb) / (s+1)
    samples = []
    samples.append(lb)
    for i in range (s):
        current += iterator
        samples.append(current)
    
    samples.append(ub)
    samples = np.array(samples)   
    return samples

# =============================================================================
# Calculate Bounds for Uncertain Attribute
# =============================================================================
def calculate_bounds(mean, w):
    lower_bound = mean - w * 0.5
    upper_bound = mean + w * 0.5
    return lower_bound, upper_bound

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

    return X


# =============================================================================
# fetch n random data tuples from a big dataset
# =============================================================================
def fetch_random_data(dataset, n, random_seed):
    np.random.seed(random_seed)  # Set the seed for reproducibility
    
    # Use the sample method directly on the dataset to get n random rows
    random_subset_indices = np.random.choice(len(dataset.data), n, replace=False)
    
    X = pd.DataFrame(dataset.data[random_subset_indices])
    y = pd.Series(dataset.target[random_subset_indices])
    
    return X, y.astype(int)

# =============================================================================
# Scale dataset to [0,1]
# =============================================================================
def normalize_dataframe(df):

    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df.values)
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns, index=df.index)
    return normalized_df
