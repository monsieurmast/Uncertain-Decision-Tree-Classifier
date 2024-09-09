# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:43:27 2024

@author: Simon Stellmach
"""

import numpy as np
import pandas as pd
import Transform_data_numerical
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
# Calculate the impurity of a set of labels
# =============================================================================

def entropy(y, probs, base):
    size = np.sum(probs)
    
    if size==0:
        return 1, 0
    
    # Use np.bincount to calculate probabilities for each unique value in y
    probabilities = np.bincount(y, weights=probs, minlength=base) / size
    
    # Avoid log(0) by adding a small value (1e-10) keep in mind
    probabilities = np.where(probabilities == 0, 1, probabilities)
    
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy, size

# =============================================================================
# Calculate information Gain for a potential split
# =============================================================================

def information_gain(parent_entropy, left_labels, left_probs, right_labels, right_probs, base):
    # Calculate entropy for the left child
    left_entropy, left_weight = entropy(left_labels, left_probs, base)
    
    # Calculate entropy for the right child
    right_entropy, right_weight = entropy(right_labels, right_probs, base)
    
    # Calculate weight of parent data
    total_samples = left_weight + right_weight
    
    # Calculate information gain
    information_gain = parent_entropy - (left_weight * left_entropy + right_weight * right_entropy) / total_samples
    
    return information_gain

# =============================================================================
# Find the best splitting value for a certain node by calculating information gain
# for every single certain attribute as well as for s samples of the distribution of
# every uncertain attribute
# =============================================================================

def find_best_split(X, y, probs, base, min_sample_split):
    best_gain = 0
    best_feature = None
    best_value = None
    parent_entropy = entropy(y, probs, base)[0]
    
    # Feature
    for feature in range (X.shape[1]):
        
        # Split Points to consider per feature
        for threshold in range (X.shape[2]):
            if np.isnan(threshold):
                break
            
            # Extracting probabilities using advanced indexing
            left_probs = X[:, feature, threshold]
            right_probs = 1 - left_probs
            left_labels, left_probs, right_labels, right_probs = split_entropy(left_probs, right_probs, y)
            
            if (len(left_labels) != 0 and len(right_labels) != 0):
                gain = information_gain(parent_entropy, left_labels, left_probs, right_labels, right_probs, base)
            
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = threshold
                
    return best_gain, best_feature, best_value

# =============================================================================
# Split labels with associated probabilities into two subsets for entropy calculation:
# =============================================================================

def split_entropy(left_probs, right_probs, y):
    left_mask = left_probs > 0
    right_mask = right_probs > 0

    left_labels = y[left_mask]
    right_labels = y[right_mask]
    left_probs = left_probs[left_mask]
    right_probs = right_probs[right_mask]
    
    return left_labels, left_probs, right_labels, right_probs

# =============================================================================
# Actually split data after optimal split was found:
# =============================================================================

def split_data(X, y, feature, threshold):
    
    left_probs = X[:, feature, threshold]
    right_probs = 1 - left_probs
    
    left_mask = left_probs > 0
    right_mask = right_probs > 0

    left_labels = y[left_mask]
    right_labels = y[right_mask]

    X_left, X_right = recalculate_probs(X, feature, threshold, left_probs, right_probs)

    return X_left[left_mask], X_right[right_mask], left_labels, right_labels, left_probs[left_mask], right_probs[right_mask]


# =============================================================================
# calculates left and right dataset after split
# =============================================================================
def recalculate_probs(X, feature, threshold, left_probs, right_probs):
    X_left = X.copy()
    X_right = X.copy()

    left_mask = left_probs != 0
    right_mask = right_probs != 0

    X_left[left_mask, feature, :] = (X[left_mask, feature, :] / left_probs[left_mask, np.newaxis])
    X_right[right_mask, feature, :] = ((X[right_mask, feature, :] - X[right_mask, feature, threshold][:, np.newaxis]) / right_probs[right_mask, np.newaxis])

    X_left[:, feature, threshold + 1:] = np.nan  # everything > threshold = nan
    X_right[:, feature, :threshold + 1] = np.nan  # everything <= threshold = nan

    return X_left, X_right


# =============================================================================
# Builds the Tree for the distribution-based approach
# =============================================================================
def build_tree(X, y, max_depth, min_samples_split, minimal_gain, probs=None, original_labels=None, original_distribution=None,):

    if original_labels is None:
        original_labels = y.copy()
    if original_distribution is None:
        original_distribution = np.ones(len(y))
    if probs is None: 
        probs = np.ones(len(y))

    if max_depth == 0 or len(np.unique(y)) == 1 or np.sum(probs) < min_samples_split:
        leaf_node = Node(data=X, labels=y, distribution=calculate_leaf_distribution(y, probs, len(np.unique(original_labels))))
        return leaf_node
    
    gain, best_feature, best_value = find_best_split(X, y, probs, len(np.unique(original_labels)), min_samples_split)
    print(gain, best_feature, best_value)
    if gain <= minimal_gain or gain == None:
        leaf_node = Node(data=X, labels=y, distribution=calculate_leaf_distribution(y, probs, len(np.unique(original_labels))))
        return leaf_node

    X_left, X_right, y_left, y_right, probs_left, probs_right = split_data(X, y, best_feature, best_value)

    left_subtree = build_tree(X_left, y_left, max_depth - 1, min_samples_split, minimal_gain, probs_left, original_labels, np.bincount(y_left, minlength=len(original_distribution)) / len(y_left))
    right_subtree = build_tree(X_right, y_right, max_depth - 1, min_samples_split, minimal_gain, probs_right, original_labels, np.bincount(y_right, minlength=len(original_distribution)) / len(y_right))

    decision_node = Node(data=X, labels=y, distribution=probs, left=left_subtree, right=right_subtree, feature_index=best_feature, threshold=best_value)

    return decision_node


# =============================================================================
# Calculate class distribution for a leaf node
# =============================================================================

def calculate_leaf_distribution(y, probs, base):
    size = 0
    probabilities = np.zeros(base)

    for i in range(len(y)):
        c = y[i]
        probabilities[c] += probs[i]
        size += probs[i]
    probabilities = probabilities / size
    return probabilities


# =============================================================================
# Predict Classification for every Tuple in the Test Data
# =============================================================================

def predict_tree(tree, X_test):
    predictions = np.empty(len(X_test), dtype=np.int32)
    for i in range(len(X_test)):
        probabilities = predict(tree, X_test[i,:,:])
        predicted_label = np.argmax(probabilities)
        predictions[i] = predicted_label
    return predictions

# =============================================================================
# Recursive Function for predicting the class of a Tuple
# =============================================================================

def predict(tree, x, weight=1):
    if tree.left is None and tree.right is None:
        # If it's a leaf node, return the normalized class distribution multiplied by the weight
        return tree.distribution * weight

    X_left, X_right, prob_left, prob_right = calculate_prob(x, tree.feature_index, tree.threshold)
    
    result_left = predict(tree.left, X_left, prob_left * weight)
    result_right = predict(tree.right, X_right, prob_right * weight)
    
    combined_results = result_left + result_right
    return combined_results



def calculate_prob(x, feature, threshold):
    
    x_left = x.copy()
    x_right = x.copy()
    
    prob_left = x[feature, threshold]
    prob_right = 1 - prob_left
    
    for i in range(x.shape[1]):  # Assuming the third dimension is the number of features
        if prob_left != 0:
            x_left[feature, i] = x[feature, i] / prob_left
        if prob_right != 0:
            x_right[feature, i] = (x[feature, i] - x[feature, threshold]) / prob_right
    
    x_left[feature, threshold + 1:] = np.nan  # everything > threshold = nan
    x_right[feature, :threshold + 1] = np.nan  # everything <= threshold = nan

    return x_left, x_right, prob_left, prob_right



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

w = 0.1
u = 0.1 
s=10
mean_based = False
perturbed = True
random_seed = 1


udf, y = Transform_data_numerical.transform_dataset(X, y, w, s, random_seed, mean_based, perturbed, u)
X_train, X_test, y_train, y_test = train_test_split(udf, y, test_size=0.2, random_state=random_seed)

tree = build_tree(X_train, y_train, max_depth, min_samples_split, minimal_gain)
y_predict = predict_tree(tree, X_test)

accuracy = accuracy_score(y_test, y_predict)
