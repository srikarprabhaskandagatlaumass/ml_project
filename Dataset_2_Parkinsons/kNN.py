# Srikar Prabhas Kandagatla (34964700)
"""
This code implements the k-NN algorithm with stratified k-fold cross-validation in Python.
The algorithm finds or predicts the class of a given unknown variable. 

This code was completed and submitted for the course COMPSCI 589: Machine Learning as part of
Homework Assignment 1, University of Massachusetts, Amherst.
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from collections import Counter

# Method for calculating the Accuracy
def calculate_precision(true_labels, predicted_labels):
    true_labels = true_labels.flatten()
    predicted_labels = np.array(predicted_labels).flatten()
    correct_predictions = np.sum(true_labels == predicted_labels)
    total_instances = len(true_labels)
    accuracy = correct_predictions / total_instances
    return accuracy

# Method to determine the majority class among k nearest neighbors
def max_count_labels(labels, indices, k):
    nearest_labels = labels[indices].flatten()
    most_common = Counter(nearest_labels).most_common(1)
    return most_common[0][0]

# Function to compute Euclidean distance between points
def compute_euclidean_distance(data_points, single_point):
    return np.sqrt(np.sum((data_points - single_point)**2, axis=1))

# Function to scale features (normalize data)
def scale_features_normalization(features, apply_scaling):
    if not apply_scaling:
        return features
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    # Handle constant features (prevent division by zero)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    return (features - min_vals) / range_vals

def stratifiedKFoldCrossValidation(X, y, k=5):
    """
    Perform stratified k-fold cross-validation.
    Splits the dataset into k folds while maintaining the class distribution in each fold.
    """
    classes = np.unique(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in classes}

    # Shuffle indices for each class
    for cls in classes:
        np.random.shuffle(class_indices[cls])

    # Create folds
    folds = [[] for _ in range(k)]
    for cls in classes:
        cls_indices = class_indices[cls]
        fold_sizes = [len(cls_indices) // k] * k
        for i in range(len(cls_indices) % k):
            fold_sizes[i] += 1
        
        current = 0
        for i in range(k):
            start, end = current, current + fold_sizes[i]
            folds[i].extend(cls_indices[start:end])
            current = end

    # Generate train-test splits
    splits = []
    for i in range(k):
        test_indices = np.array(folds[i])
        train_indices = np.concatenate([folds[j] for j in range(k) if j != i])
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        splits.append((X_train, y_train, X_test, y_test))

    return splits

def k_nearest_neighbor_classifier(data, neighbors=3, scale_data=True, use_cross_val=True):
    # Prepare the data
    feature_count = data.shape[1] - 1
    X, y = data[:, :feature_count], data[:, -1:]
    
    if use_cross_val:
        # Perform stratified k-fold cross-validation
        splits = stratifiedKFoldCrossValidation(X, y, k=10)
        
        train_accuracies = []
        test_accuracies = []
        
        for X_train, y_train, X_test, y_test in splits:
            # Scale features if needed
            X_train_scaled = scale_features_normalization(X_train, scale_data)
            X_test_scaled = scale_features_normalization(X_test, scale_data) if scale_data else X_test
            
            # Predictions for training set
            train_predictions = []
            for i, x in enumerate(X_train_scaled):
                if neighbors == 1:
                    train_predictions.append(y_train[i])
                else:
                    distances = compute_euclidean_distance(X_train_scaled, x)
                    nearest_indices = np.argsort(distances)[1:neighbors+1]
                    pred = max_count_labels(y_train, nearest_indices, neighbors)
                    train_predictions.append(pred)
            
            # Predictions for test set
            test_predictions = []
            for x in X_test_scaled:
                distances = compute_euclidean_distance(X_train_scaled, x)
                nearest_indices = np.argsort(distances)[:neighbors]
                pred = max_count_labels(y_train, nearest_indices, neighbors)
                test_predictions.append(pred)
            
            train_acc = calculate_precision(y_train, train_predictions)
            test_acc = calculate_precision(y_test, test_predictions)
            
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
        
        return np.mean(train_accuracies), np.mean(test_accuracies)
    else:
        # Original train-test split implementation
        shuffled_data = shuffle(data)
        X, y = shuffled_data[:, :feature_count], shuffled_data[:, -1:]
        scaled_X = scale_features_normalization(X, scale_data)
        
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)
        
        # Predictions for training set
        train_predictions = []
        for i, x in enumerate(X_train):
            if neighbors == 1:
                train_predictions.append(y_train[i])
            else:
                distances = compute_euclidean_distance(X_train, x)
                nearest_indices = np.argsort(distances)[1:neighbors+1]
                pred = max_count_labels(y_train, nearest_indices, neighbors)
                train_predictions.append(pred)
        
        # Predictions for test set
        test_predictions = []
        for x in X_test:
            distances = compute_euclidean_distance(X_train, x)
            nearest_indices = np.argsort(distances)[:neighbors]
            pred = max_count_labels(y_train, nearest_indices, neighbors)
            test_predictions.append(pred)
        
        train_precision = calculate_precision(y_train, train_predictions)
        test_precision = calculate_precision(y_test, test_predictions)
        
        return train_precision, test_precision

# Method to plot results
def plot_results(k_values, accuracies, std_devs, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, accuracies, yerr=std_devs, fmt='-D', capsize=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.xlim(0, 52)
    plt.xticks(k_values)
    plt.show()

# ----Main Execution-----
# Loading the dataset
dataset = pd.read_csv('Dataset_2_Parkinsons/parkinsons.csv', header=0).values

# Define range of k values and number of iterations
k_range = range(1, 52, 2)
iterations = 20

# Initialize results dictionary
results = {
    'normalized': {'training': [], 'testing': []},
    'non_normalized': {'training': [], 'testing': []}
}

# Run k-NN for different k values and with/without normalization
for scaling in [True, False]:
    key = 'normalized' if scaling else 'non_normalized'
    for k in k_range:
        train_acc, test_acc = [], []
        for _ in range(iterations):
            train_prec, test_prec = k_nearest_neighbor_classifier(
                dataset, k, scaling, use_cross_val=True)
            train_acc.append(train_prec)
            test_acc.append(test_prec)
        results[key]['training'].append((np.mean(train_acc), np.std(train_acc)))
        results[key]['testing'].append((np.mean(test_acc), np.std(test_acc)))

# Plot results
for data_type in ['normalized', 'non_normalized']:
    for set_type in ['training', 'testing']:
        means, stds = zip(*results[data_type][set_type])
        plot_results(k_range, means, stds,
                     f'k-NN {set_type.capitalize()} Set ({data_type.capitalize()} Data)\n(Stratified 10-Fold CV)',
                     'Value of K', f'Accuracy Over {set_type.capitalize()} Data')

# Printing accuracies for all values of 'k' with decimal places
for data_type in ['normalized', 'non_normalized']:
    print(f"\nAccuracies for {data_type} data:")
    for k, (train_acc, test_acc) in zip(k_range, zip(results[data_type]['training'], results[data_type]['testing'])):
        print(f"k = {k}:")
        print(f"Training accuracy: {train_acc[0]:.4f}, {train_acc[1]:.4f}")
        print(f"Testing accuracy: {test_acc[0]:.4f}, {test_acc[1]:.4f}")

# Printing maximum testing accuracies
print("\nMaximum testing accuracies:")
for data_type in ['normalized', 'non_normalized']:
    test_accuracies = [acc[0] for acc in results[data_type]['testing']]
    max_test_acc = max(test_accuracies)
    max_test_k = k_range[test_accuracies.index(max_test_acc)]
    print(f"{data_type.capitalize()} data: Maximum accuracy = {max_test_acc:.4f} at k = {max_test_k}")