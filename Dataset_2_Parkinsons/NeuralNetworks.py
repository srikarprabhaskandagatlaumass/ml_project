# Srikar Prabhas Kandagatla (34964700)
"""
This code implements the Neural Network in Python to classify WDBC, Loan,
Titanic, and Raisins dataset.

I executed the code using the VS Code IDE. To run it on your local machine, download the
dataset (four datasets), set the working directory, and execute the script.

This code was completed and submitted as part of Homework Assignment 4 for
COMPSCI 589: Machine Learning at the University of Massachusetts, Amherst for Spring'25 Semester.
"""

# COMPSCI 589: Machine Learning (Spring 2025)
"""
In this code, I have implemented the Neural Network Architecture to classify the WDBC, Loan,
Titanic, and Raisins datasets. This model also supports the Numeric and Categorical features which
can support varaible architectures.
"""

# Importing the Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Helper Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoidDerivative(a):
    return a * (1 - a)

def addBiasNeuron(X):
    if X.ndim == 1:
        return np.concatenate([[1], X])
    return np.column_stack([np.ones(X.shape[0]), X])

def generateInitialWeights(layer_sizes):
    adjusted_weights = []
    for i in range(len(layer_sizes) - 1):
        weight_matrix = np.random.uniform(low=-1, high=1, size=(layer_sizes[i + 1], layer_sizes[i] + 1))
        adjusted_weights.append(weight_matrix)  
    return adjusted_weights

# Forward and Backward Propagation Functions to calculate the cost and gradients_regularizationients in the Neural Network done trainModel function
def forwardPropagation(X, weights):
    a_values = [addBiasNeuron(X)] 
    z_values = []

    for i, theta in enumerate(weights):
        z = np.dot(a_values[-1], theta.T)
        z_values.append(z)
        a = sigmoid(z)

        if i < len(weights) - 1: 
            a = addBiasNeuron(a) 
        a_values.append(a)

    return a_values, z_values

def calculateCost(X, y, weights, lambda_regulation):
    m = X.shape[0]  
    a_values, _ = forwardPropagation(X, weights)
    a_final = a_values[-1]
    a_final = np.clip(a_final, 1e-10, 1 - 1e-10)

    cost_per_example = y * (-np.log(a_final)) + (1 - y) * (-np.log(1 - a_final))
    total_cost = np.sum(cost_per_example) / m  

    if m > 1: 
        regularization = 0
        for theta in weights:
            regularization += np.sum(theta[:, 1:]**2)  
        regularization = (lambda_regulation / (2 * m)) * regularization
        total_cost += regularization

    return total_cost

def backwardPropagation(X, y, weights, lambda_regulation):
    m = X.shape[0] 
    a_values, _ = forwardPropagation(X, weights)
    deltas = [a_values[-1] - y]  

    for l in range(len(weights) - 1, 0, -1):
        if a_values[l].ndim == 1: 
            delta = np.dot(deltas[-1], weights[l][:, 1:]) * sigmoidDerivative(a_values[l][1:])
        else: 
            delta = np.dot(deltas[-1], weights[l][:, 1:]) * sigmoidDerivative(a_values[l][:, 1:])
        deltas.append(delta)

    deltas.reverse()

    gradients_regularizationients = []
    for l in range(len(weights)):
        gradients_regularization = np.dot(deltas[l].T, a_values[l]) / m
        
        if lambda_regulation != 0:
            regularization = (lambda_regulation / m) * weights[l]
            regularization[:, 0] = 0 
            gradients_regularization += regularization
        
        gradients_regularizationients.append(gradients_regularization)

    return gradients_regularizationients

def generateMiniBatches(X, y, batch_size):
    m = X.shape[0]
    indices = np.random.permutation(m) 
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    mini_batches = []
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, y_batch))

    return mini_batches

# Compleented the epsilon stopping creiteria and implemented the maximum iterations, no commeent and change the code loop to while loop to use epsilion stopping criteria
def trainModel(X, y, weights, alpha_learning_rate=0.5, lambda_regulation=0.0, max_iterations=1, batch_size=1, epsilon=1e-6):
    y = np.array(y).reshape(-1, 1)  # Ensure labels are binary and reshaped correctly
    
    for iteration in range(max_iterations):
        mini_batches = generateMiniBatches(X, y, batch_size)
        total_cost = 0
        accumulated_gradients = [np.zeros_like(w) for w in weights] 
        
        for X_batch, y_batch in mini_batches:
            gradients = backwardPropagation(X_batch, y_batch, weights, 0) 
            cost = calculateCost(X_batch, y_batch, weights, lambda_regulation)
            total_cost += cost
            
            for l in range(len(weights)):
                accumulated_gradients[l] += gradients[l] * len(X_batch)

        m = X.shape[0]
        for l in range(len(accumulated_gradients)):
            avg_gradients = accumulated_gradients[l] / m
            avg_gradients += (lambda_regulation / m) * weights[l]
            avg_gradients[:, 0] = accumulated_gradients[l][:, 0] / m 
            
            weights[l] -= alpha_learning_rate * avg_gradients
        
        total_cost /= len(mini_batches)
        print(f"Iteration: {iteration+1}, Cost: {total_cost:.5f}")
    
    return weights, gradients

def dataNormalization(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    X_range = X_max - X_min
    X_range[X_range == 0] = 1 
    
    X_normalized = 2 * (X - X_min) / X_range - 1 # [-1, +1]

    return X_normalized, X_min, X_max

# def preprocessDataset(dataset_path):
#     data = pd.read_csv(dataset_path)

#     column_names = data.columns.tolist()

#     if 'label' in column_names:
#         label_column = 'label'
#     else:
#         label_column = column_names[-1] # Default to the last column

#     X = data.drop(columns=[label_column]).copy()
#     y = data[label_column].values # Label column

#     categorical_columns = [col for col in X.columns if '_cat' in col]
#     numerical_columns = [col for col in X.columns if '_num' in col]

#     if categorical_columns:
#         encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#         X_categorical_encoded = encoder.fit_transform(X[categorical_columns])
#         X_categorical_encoded = pd.DataFrame(X_categorical_encoded, index=X.index)
#     else:
#         X_categorical_encoded = pd.DataFrame(index=X.index) # Empty DataFrame if no categorical columns

#     X_numerical = X[numerical_columns]
#     X_processed = pd.concat([X_numerical, X_categorical_encoded], axis=1).values
#     X_normalized, _, _ = dataNormalization(X_processed)

#     y = y.reshape(-1, 1)

#     return X_normalized, y

def preprocessDataset(dataset_path):
    data = pd.read_csv(dataset_path)

    # Always take the last column as label, all others as features
    X = data.iloc[:, :-1].values   # All rows, all columns except last
    y = data.iloc[:, -1].values    # All rows, last column only

    # If y is categorical and you need it as numeric, convert here if needed
    # y = pd.to_numeric(y, errors='coerce')

    # Normalize features (optional, but common)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_normalized = 2 * (X - X_min) / X_range - 1

    y = y.reshape(-1, 1)
    return X_normalized, y

def calculateEvaluationMetrics(y_true, y_pred):
    y_pred_limit = (y_pred >= 0.5).astype(int)

    TP = np.sum((y_true == 1) & (y_pred_limit == 1))
    FP = np.sum((y_true == 0) & (y_pred_limit == 1))
    TN = np.sum((y_true == 0) & (y_pred_limit == 0))
    FN = np.sum((y_true == 1) & (y_pred_limit == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score

def predictClass(X, weights):
    a_values, _ = forwardPropagation(X, weights)
    y_pred = a_values[-1] 
    return y_pred

def stratifiedKFoldCrossValidation(X, y, k=5):
    classes = np.unique(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in classes}

    for cls in classes:
        np.random.shuffle(class_indices[cls])

    folds = [[] for _ in range(k)]
    for cls in classes:
        cls_indices = class_indices[cls]
        cls_folds = np.array_split(cls_indices, k)
        for i in range(k):
            folds[i].extend(cls_folds[i])

    splits = []
    for i in range(k):
        test_indices = np.array(folds[i])
        train_indices = np.array([idx for fold in folds if fold != folds[i] for idx in fold])
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        splits.append((X_train, y_train, X_test, y_test))

    return splits

def evaluateModelAnalysis(X, y, k):
    """
    Evaluate the model using stratified k-fold cross-validation.
    Takes the dataset (X, y) as input.
    """
    # Normalize the dataset
    X, _, _ = dataNormalization(X)

    # Updated configurations
    architectures = [
        [X.shape[1], 128, 64, 1],
        [X.shape[1], 4, 1],
        [X.shape[1], 1],  # Output layer has 1 neuron for binary classification
        [X.shape[1], 64, 1],
        [X.shape[1], 128, 1]
    ]

    lambda_values = [0.0, 0.01, 0.001]
    learning_rates = [0.1, 0.05]
    
    results = []

    # Perform stratified k-fold cross-validation
    splits = stratifiedKFoldCrossValidation(X, y, k=k)

    for arch in architectures:
        for lr in learning_rates:
            for lmbda in lambda_values:
                print(f"\nTesting: {arch}, lambda={lmbda}, alpha={lr}")

                fold_accuracies = []
                fold_f1_scores = []

                for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(splits):
                    print(f"  Fold {fold_idx + 1}/{k}")

                    # Initialize weights
                    weights = generateInitialWeights(arch)

                    # Train the model
                    trained_weights, _ = trainModel(
                        X_train, y_train,  # Use raw binary labels
                        weights=weights,
                        alpha_learning_rate=lr,
                        lambda_regulation=lmbda,
                        max_iterations=50,  # Increased
                        batch_size=32        # Increased
                    )

                    # Predict and evaluate
                    y_pred = predictClass(X_test, trained_weights)
                    accuracy, precision, recall, f1_score = calculateEvaluationMetrics(y_test, y_pred)

                    fold_accuracies.append(accuracy)
                    fold_f1_scores.append(f1_score)

                # Average metrics across folds
                avg_accuracy = np.mean(fold_accuracies)
                avg_f1_score = np.mean(fold_f1_scores)

                # Store results
                results.append({
                    "architecture": arch,
                    "lambda": lmbda,
                    "learning_rate": lr,
                    "avg_accuracy": avg_accuracy,
                    "avg_f1": avg_f1_score
                })

    # Print results
    print("\nResults Summary:")
    print("=" * 90)
    print(f"{'Architecture':<25} | {'Lambda':<8} | {'LR':<6} | {'Accuracy':<10} | {'F1':<10}")
    print("=" * 90)

    sorted_results = sorted(results, key=lambda x: (x['avg_accuracy'], x['avg_f1']), reverse=True)
    
    for res in sorted_results:
        print(
            f"{str(res['architecture']):<25} | "
            f"{res['lambda']:<8.4f} | "
            f"{res['learning_rate']:<6.3f} | "
            f"{res['avg_accuracy']:.4f}     | "
            f"{res['avg_f1']:.4f}"
        )
    
    return results

if __name__ == "__main__":
    case = None # Set to 1 for the first test case, 2 for the second test case, or None to skip test cases

    dataset_path = "Dataset_2_Parkinsons/parkinsons.csv" # Change this for using different datasets
    X, y = preprocessDataset(dataset_path)

    indices = np.random.permutation(X.shape[0]) 
    X = X[indices] 
    y = y[indices] 

    layer_sizes = [X.shape[1], 16, 2]
    alpha_learning_rate=0.05
    lambda_regulation=0.1
    max_iterations=1000
    batch_size=32
    # epsilon=0.00001
    k=10

    if case is not None:
        evaluateModelAnalysis(X, y, k)
    else:
        dataset_split_ratio = 0.8 
        dataset_split_index = int(dataset_split_ratio * X.shape[0])
        X_train, X_test = X[:dataset_split_index], X[dataset_split_index:]
        y_train, y_test = y[:dataset_split_index], y[dataset_split_index:]

        training_sizes = []
        test_costs = []

        weights = generateInitialWeights(layer_sizes)

        for samples in range(5, X_train.shape[0] + 1, 5): 
            print(f"\nTraining with {samples} samples.")

            X_train_subset = X_train[:samples]
            y_train_subset = y_train[:samples]

            trained_weights, gradients_regularizationients = trainModel(
                X_train_subset, y_train_subset,
                weights=weights,
                alpha_learning_rate=alpha_learning_rate,
                lambda_regulation=lambda_regulation,
                max_iterations=max_iterations,
                batch_size=batch_size,
                # epsilon=epsilon
            )

            test_cost = calculateCost(X_test, y_test, trained_weights, lambda_regulation)
            print(f"Test Cost (J) with {samples} training samples: {test_cost:.5f}")

            training_sizes.append(samples)
            test_costs.append(test_cost)
        
        dataset_title = (dataset_path.split("/")[-1].split(".")[0]).capitalize() 

        plt.figure(figsize=(10, 6))
        plt.plot(training_sizes, test_costs, label="Cost (J)", linewidth=2)
        plt.title(f"Learning Curve - NN: {layer_sizes} - {dataset_title} Dataset - Lambda: {lambda_regulation} - Alpha: {alpha_learning_rate} - Batch Size: {batch_size}")
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Cost Function (J)")
        plt.legend()
        plt.show()