# Srikar Prabhas Kandagatla (34964700)
"""
This code implements the k-NN algorithm in Python. The algorithm finds or predicts
the class of a given unknown variable. The overall execution time is usually under a minute.

I have executed the following code in Google Colab with CPU runtime type. Please feel
free to use this code. It can be executed in any IDE that has the required packages installed
(Numpy, Pandas, Matplotlib, and scikit-learn).

This code was completed and submitted for the course COMPSCI 589: Machine Learning as part of
Homework Assignment 1, University of Massachusetts, Amherst.
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Hand Writtern Dataset
# digits = datasets.load_digits(return_X_y=True)
# digits_dataset_X = digits[0]
# digits_dataset_y = digits[1]

# Load digits dataset and combine features + labels
digits = datasets.load_digits()
digits_dataset_X = digits.data
digits_dataset_y = digits.target.reshape(-1, 1)
digits_combined = np.hstack((digits_dataset_X, digits_dataset_y))  # Combine features and labels

# Use in main execution
dataset = digits_combined  # Now shaped (1797, 65) - 64 pixels + 1 label column

# Method for calculating the Accuracy
def calculate_precision(true_labels, predicted_labels):
    true_labels = true_labels.flatten()
    predicted_labels = np.array(predicted_labels).flatten()
    correct_predictions = np.sum(true_labels == predicted_labels)
    total_instances = len(true_labels)
    accuracy = correct_predictions / total_instances
    return accuracy
    """
    This method calculates the accuracy of the model by computing the
    percentage of correct predictions made by the model when applied to
    the testing data
    """

# Method to determine the majority class among k nearest neighbors
def max_count_labels(labels, indices, k):
    nearest_labels = labels[indices].flatten()
    return int(np.sum(nearest_labels) > k / 2)
    """This medtod determines the majority of the class among the k nearest
    neighbours in the dataset"""

# Function to compute Euclidean distance between points
def compute_euclidean_distance(data_points, single_point):
    return np.sqrt(np.sum((data_points - single_point)**2, axis=1))
    """
    The Euclidean distance is calculated between the points in the dataset
    """

# Function to scale features (normalize data)
def scale_features_normalization(features, apply_scaling):
    if not apply_scaling:
        return features
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    
    # Handle constant features (prevent division by zero)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Set zero ranges to 1
    
    return (features - min_vals) / range_vals

    """
    Applying normalization or feature scaling to the training set to scale
    the values between [-1, +1]
    """

# k-Nearest Neighbors classifier method
def k_nearest_neighbor_classifier(data, neighbors=3, scale_data=True):
    # Prepare and preprocessing the data
    shuffled_data = shuffle(data)
    feature_count = data.shape[1] - 1
    X, y = shuffled_data[:, :feature_count], shuffled_data[:, -1:]
    scaled_X = scale_features_normalization(X, scale_data)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)

    # Method to predict labels
    def predict_labels(X_set, is_training=True):
        predictions = []
        for i, x in enumerate(X_set):
            if is_training and neighbors == 1:
                predictions.append(y_train[i])
            else:
                distances = compute_euclidean_distance(X_train, x)
                if is_training:
                    nearest_indices = np.argsort(distances)[1:neighbors+1]
                else:
                    nearest_indices = np.argsort(distances)[:neighbors]
                predictions.append(max_count_labels(y_train, nearest_indices, neighbors))
        return predictions

    # Performing Predictions and calculating Accuracy
    train_predictions = predict_labels(X_train)
    test_predictions = predict_labels(X_test, is_training = False)

    train_precision = calculate_precision(y_train, train_predictions)
    test_precision = calculate_precision(y_test, test_predictions)

    return train_precision, test_precision
    """
    This KNN classifier shuffles the dataset, scales features if needed,
    and splits it into training and testing sets. It predicts labels by
    computing Euclidean distances and selecting the most common label among
    the nearest neighbors. Finally, it calculates and returns the accuracy
    for both training and test sets.
    """

# Method to plot results
def plot_results(k_values, accuracies, std_devs, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, accuracies, yerr = std_devs, fmt='-D', capsize = 5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.xlim(0, 52)
    plt.xticks(k_values)
    plt.show()
    """
    This method plots results of k-NN classifier by displaying accuracy with
    error bars for different values of 'k'
    """


# ----Main Execution-----
# Loading the dataset
# dataset = pd.read_csv('wdbc.csv', header=None).values
# dataset = digits

# Define range of k values and number of iterations (User Defined)
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
            train_prec, test_prec = k_nearest_neighbor_classifier(dataset, k, scaling)
            train_acc.append(train_prec)
            test_acc.append(test_prec)
        results[key]['training'].append((np.mean(train_acc), np.std(train_acc)))
        results[key]['testing'].append((np.mean(test_acc), np.std(test_acc)))
"""
This chuck evaluates a KNN classifier with and without feature scaling.
It iterates over two cases: normalized and non-normalized data. For each value of
k, it runs multiple iterations, stores the training and testing accuracy, and
computes their mean and standard deviation. The results are saved in a
dictionary for further analysis.
"""

# Plot results
for data_type in ['normalized', 'non_normalized']:
    for set_type in ['training', 'testing']:
        means, stds = zip(*results[data_type][set_type])
        plot_results(k_range, means, stds,
                     f'k-NN {set_type.capitalize()} Set ({data_type.capitalize()} Data)',
                     'Value of K', f'Accuracy Over {set_type.capitalize()} Data')

# Printing accuracies for all values of 'k' with decimal places
for data_type in ['normalized', 'non_normalized']:
    print(f"\nAccuracies for {data_type} data:")
    for k, (train_acc, test_acc) in zip(k_range, zip(results[data_type]['training'], results[data_type]['testing'])):
        print(f"k = {k}:")
        print(f"Training accuracy: {train_acc[0]:.4f} ± {train_acc[1]:.4f}")
        print(f"Testing accuracy: {test_acc[0]:.4f} ± {test_acc[1]:.4f}")

    print(f"\nMaximum accuracies for {data_type} data:")

# Printing maximum testing accuracies for Normalized and Non Normalized
print("\nMaximum testing accuracies:")
for data_type in ['normalized', 'non_normalized']:
    test_accuracies = [acc[0] for acc in results[data_type]['testing']]
    max_test_acc = max(test_accuracies)
    max_test_k = k_range[test_accuracies.index(max_test_acc)]
    print(f"{data_type.capitalize()} data: Maximum accuracy = {max_test_acc:.4f} at k = {max_test_k}")