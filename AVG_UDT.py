# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:43:27 2024

@author: Simon Stellmach
"""

import Transform_data_avg_udt

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =============================================================================
# Node class necessary to start building the tree
# =============================================================================

class Node:
    def __init__(self, data, labels, distribution=None, left=None, right=None, feature_index=None, threshold=None):
        self.data = data
        self.labels = labels
        self.distribution = distribution
        self.left = left
        self.right = right
        self.feature_index = feature_index
        self.threshold = threshold
        

# =============================================================================
# 
# Training functions
# 
# =============================================================================
# =============================================================================
# Calculate the impurity of a set of labels
# =============================================================================

def entropy(y, base):
    size = len(y)
    # Count occurrences of each class
    class_counts = np.bincount(y, minlength=base)
    # Calculate probabilities
    probabilities = class_counts / size
    # Avoid log(0) by adding a small value (1e-10)
    probabilities = np.maximum(probabilities, 1e-10)

    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy, size

# =============================================================================
# Calculate information Gain for a potential split
# =============================================================================

def information_gain(parent_entropy, left_labels, right_labels, base):
    left_size = len(left_labels)
    right_size = len(right_labels)

    left_probs = np.bincount(left_labels, minlength=base) / left_size
    right_probs = np.bincount(right_labels, minlength=base) / right_size

    # Avoid log(0) by adding a small value
    left_probs = np.maximum(left_probs, 1e-10)
    right_probs = np.maximum(right_probs, 1e-10)

    left_entropy = -np.sum(left_probs * np.log2(left_probs))
    right_entropy = -np.sum(right_probs * np.log2(right_probs))

    information_gain = parent_entropy - (left_size * left_entropy +
                                         right_size * right_entropy) / (left_size + right_size)

    return information_gain


# =============================================================================
# Find the best splitting value for a certain node by calculating information gain
# for every single certain attribute as well as for s samples of the distribution of
# every uncertain attribute
# =============================================================================

def find_best_split(X, y, base, min_sample_split):
    num_features = X.shape[1]
    best_gain = 0
    best_feature = None
    best_value = None
    
    max_value = 4
    
    for feature in range(0, num_features, max_value):
        means = X[:, feature]
        split_points = np.unique(means)
        
        for i in range(len(split_points) - 1):
            point = (split_points[i] + split_points[i + 1]) / 2 
            left_indices = means <= point
            right_indices = ~left_indices
            left_labels, right_labels = y[left_indices], y[right_indices]
            
            gain = information_gain(entropy(y, base)[0], left_labels, right_labels, base)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_value = point

    print(best_gain, best_feature / max_value if best_feature is not None else 'None', best_value)                        
    return best_gain, best_feature, best_value

# =============================================================================
# Build the Tree recursively by calculating the best split for every node of the Decision Tree
# =============================================================================

def build_tree(X, y, max_depth, min_samples_split, minimal_gain, original_labels=None, original_distribution=None):
    if original_labels is None:
        original_labels = y.copy()
    if original_distribution is None:
        original_distribution = np.bincount(y, minlength=len(original_labels)) / len(y)

    if max_depth == 0 or len(np.unique(y)) == 1 or len(y) < min_samples_split:
        leaf_node = Node(data=X, labels=y, distribution=calculate_leaf_distribution(y, len(np.unique(original_labels))))
        return leaf_node

    gain, best_feature, best_value = find_best_split(X, y, len(np.unique(original_labels)), min_samples_split)
    if gain < minimal_gain:
        leaf_node = Node(data=X, labels=y, distribution=calculate_leaf_distribution(y, len(np.unique(original_labels))))
        return leaf_node

    X_left, X_right, y_left, y_right = split_data(X, y, best_feature, best_value, True)

    left_subtree = build_tree(X_left, y_left, max_depth - 1, min_samples_split, minimal_gain, original_labels, original_distribution)
    right_subtree = build_tree(X_right, y_right, max_depth - 1, min_samples_split, minimal_gain, original_labels, original_distribution)

    decision_node = Node(data=X, labels=y, distribution=original_distribution, left=left_subtree, right=right_subtree, feature_index=best_feature, threshold=best_value)

    return decision_node

# =============================================================================
# Calculate class distribution for a leaf node
# =============================================================================

def calculate_leaf_distribution_mean(y, base):
    size = len(y)
    probabilities = np.zeros(base)
    for i in range(len(y)):
        c = y[i]
        probabilities[c] += 1
    probabilities = probabilities / size
    return probabilities

# =============================================================================
# Split labels with associated probabilities into two subsets:
# -> both = True also returns the split sets of X - used for actually splitting dataset
# -> both = False only returns labels and associated probs for calculating information gain after split
# =============================================================================

def split_data(X, y, feature, threshold, both):
    left_indices = X[:, feature] <= threshold
    right_indices = ~left_indices

    if both:
        return X[left_indices], X[right_indices], y[left_indices], y[right_indices]
    else:
        return y[left_indices], y[right_indices]
    
# =============================================================================
# Calculate class distribution for a leaf node
# =============================================================================

def calculate_leaf_distribution(y, base):
    size = len(y)
    probabilities = np.zeros(base)
    for i in range(len(y)):
        c = y[i]
        probabilities[c] += 1
    probabilities = probabilities / size
    return probabilities

# =============================================================================
# 
# PREDICTION
#
# =============================================================================
# =============================================================================
# Recursive Function for predicting the class of a Tuple
# =============================================================================

def predict(tree, X, weight=1):
    if tree.left is None and tree.right is None:
        # If it's a leaf node, return the normalized class distribution multiplied by the weight
        return tree.distribution * weight

    X_left, X_right, prob_left, prob_right = calculate_prob(X, tree.feature_index, tree.threshold)
    
    result_left = predict(tree.left, X_left, prob_left * weight)
    result_right = predict(tree.right, X_right, prob_right * weight)
    
    combined_results = result_left + result_right
    return combined_results


# =============================================================================
# Predict Classification for every Tuple in the Test Data
# =============================================================================

def predict_tree(tree, X_test):
    predictions = np.empty(len(X_test), dtype=np.int32)
    for i in range(len(X_test)):
        probabilities = predict(tree, X_test[i])
        predicted_label = np.argmax(probabilities)
        predictions[i] = predicted_label
    return predictions

# =============================================================================
# Calculate left and right probability for a certain node in a decision tree classifier
# =============================================================================

def calculate_prob(X, feature, threshold):
    mean = X[feature]
    std = X[feature + 1]
    lower_bound = X[feature + 2]
    upper_bound = X[feature + 3]

    X_left = X.copy()
    X_right = X.copy()

    if upper_bound <= threshold:
        prob_left = 1
    elif threshold < lower_bound:
        prob_left = 0
    else:
        probs = norm.cdf(upper_bound, loc=mean, scale=std) - norm.cdf(lower_bound, loc=mean, scale=std)
        prob_left = (norm.cdf(threshold, loc=mean, scale=std) - norm.cdf(lower_bound, loc=mean, scale=std)) / probs
        X_left[feature + 3] = threshold
        X_right[feature + 2] = threshold

    prob_right = 1 - prob_left

    return X_left, X_right, prob_left, prob_right





# =============================================================================
#
# RUN
#
# =============================================================================

# Diabetes Dataset
diabetes = pd.read_csv("Data/diabetes.csv")
y = diabetes["Outcome"]
X = diabetes.drop(columns="Outcome")
minimal_gain = 0.01
max_depth = 3
min_samples_split = 5

# =============================================================================
# # California-Housing Dataset
# california_housing = fetch_california_housing()
# X, y = Transform_data_numerical.fetch_random_data(dataset=california_housing, n=500, random_seed=1)
# minimal_gain = 0.01
# max_depth = 5
# min_samples_split = 10
# =============================================================================

u=0.1
w=0.1
random_seed = 1

X, y = Transform_data_avg_udt.adapt_df(X, y, w, u, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

tree = build_tree(X=X_train, y=y_train, max_depth=max_depth, min_samples_split=min_samples_split, minimal_gain=minimal_gain)
y_predict = predict_tree(tree, X_test)

accuracy = accuracy_score(y_test, y_predict)
