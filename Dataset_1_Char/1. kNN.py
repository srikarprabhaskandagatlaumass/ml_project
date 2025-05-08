import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from collections import Counter

# Load digits dataset
digits = datasets.load_digits()
digits_dataset_X = digits.data
digits_dataset_y = digits.target.reshape(-1, 1)
dataset = np.hstack((digits_dataset_X, digits_dataset_y))

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

# Function to compute Euclidean distance between points (optimized)
def compute_euclidean_distance(data_points, single_point):
    return np.sqrt(np.sum((data_points - single_point)**2, axis=1))

# Function to scale features using standardization (better for KNN)
def scale_features_standardization(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (features - mean) / std

def stratifiedKFoldCrossValidation(X, y, k=5):
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

def k_nearest_neighbor_with_kfold(dataset, k_range, num_folds=5):
    X, y = dataset[:, :-1], dataset[:, -1]
    
    # Standardize all features first (z-score normalization)
    X = scale_features_standardization(X)
    
    results = {
        'training': [],
        'testing': []
    }

    for k in k_range:
        fold_train_acc, fold_test_acc = [], []

        splits = stratifiedKFoldCrossValidation(X, y, k=num_folds)
        for X_train, y_train, X_test, y_test in splits:
            # KNN prediction for test set
            test_predictions = []
            for x in X_test:
                distances = compute_euclidean_distance(X_train, x)
                nearest_indices = np.argsort(distances)[:k]
                pred = max_count_labels(y_train, nearest_indices, k)
                test_predictions.append(pred)
            
            # KNN prediction for training set (leave-one-out when k > 1)
            train_predictions = []
            for i, x in enumerate(X_train):
                if k == 1:
                    train_predictions.append(y_train[i])
                else:
                    distances = compute_euclidean_distance(X_train, x)
                    # Exclude the point itself
                    nearest_indices = np.argsort(distances)[1:k+1]
                    pred = max_count_labels(y_train, nearest_indices, k)
                    train_predictions.append(pred)
            
            train_acc = calculate_precision(y_train, train_predictions)
            test_acc = calculate_precision(y_test, test_predictions)
            
            fold_train_acc.append(train_acc)
            fold_test_acc.append(test_acc)

        results['training'].append((np.mean(fold_train_acc), np.std(fold_train_acc)))
        results['testing'].append((np.mean(fold_test_acc), np.std(fold_test_acc)))

    return results

def plot_separate_results(results, k_range):
    train_means = [x[0] for x in results['training']]
    train_stds = [x[1] for x in results['training']]
    test_means = [x[0] for x in results['testing']]
    test_stds = [x[1] for x in results['testing']]

    # Plot for Training Accuracy
    plt.figure(figsize=(12, 6))
    plt.errorbar(k_range, train_means, yerr=train_stds, fmt='-o', 
                 label='Training Accuracy', capsize=5)
    plt.xlabel('Value of K')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.grid(True)
    plt.xticks(k_range)
    plt.legend()
    plt.show()

    # Plot for Testing Accuracy
    plt.figure(figsize=(12, 6))
    plt.errorbar(k_range, test_means, yerr=test_stds, fmt='-s', label='Testing Accuracy', capsize=5)
    plt.xlabel('Value of K')
    plt.ylabel('Accuracy')
    plt.title('Testing Accuracy')
    plt.grid(True)
    plt.xticks(k_range)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Define range of k values and number of folds
    k_range = range(1, 53, 2)  # Optimal range for digits dataset
    num_folds = 10

    # Perform k-NN with stratified k-fold cross-validation
    results = k_nearest_neighbor_with_kfold(dataset, k_range, num_folds=num_folds)

    # Plot the results separately
    plot_separate_results(results, k_range)

    # Print accuracies
    print("\nAccuracies for all values of k:")
    for k, (train_acc, test_acc) in zip(k_range, zip(results['training'], results['testing'])):
        print(f"k = {k}:")
        print(f"  Training accuracy: {train_acc[0]:.4f}, {train_acc[1]:.4f}")
        print(f"  Testing accuracy: {test_acc[0]:.4f}, {test_acc[1]:.4f}")

    # Find and print maximum testing accuracy
    test_accuracies = [acc[0] for acc in results['testing']]
    max_test_acc = max(test_accuracies)
    max_test_k = k_range[test_accuracies.index(max_test_acc)]
    print(f"\nMaximum testing accuracy = {max_test_acc:.4f} at k = {max_test_k}")