# Importing the Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets

# Hand Writtern Dataset
digits = datasets.load_digits(return_X_y=True)
digits_dataset_X = digits[0]
digits_dataset_y = digits[1]

# Helper Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoidDerivative(a):
    return a * (1 - a)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def addBiasNeuron(X):
    if X.ndim == 1:
        return np.concatenate([[1], X])
    return np.column_stack([np.ones(X.shape[0]), X])

def generateInitialWeights(layer_sizes):
    adjusted_weights = []
    for i in range(len(layer_sizes) - 1):
        limit = np.sqrt(2 / layer_sizes[i]) 
        weight_matrix = np.random.uniform(low=-limit, high=limit, size=(layer_sizes[i + 1], layer_sizes[i] + 1))
        adjusted_weights.append(weight_matrix)
    return adjusted_weights

# Forward and Backward Propagation Functions to calculate the cost and gradients_regularizationients in the Neural Network done trainModel function
def forwardPropagation(X, weights):
    a_values = [addBiasNeuron(X)]
    z_values = []
    
    for i, theta in enumerate(weights):
        z = np.dot(a_values[-1], theta.T)
        z_values.append(z)
        # Use softmax for output layer
        a = softmax(z) if i == len(weights)-1 else sigmoid(z)  
        if i < len(weights)-1:
            a = addBiasNeuron(a)
        a_values.append(a)
    
    return a_values, z_values

def calculateCost(X, y, weights, lambda_regulation):
    m = X.shape[0]
    a_values, _ = forwardPropagation(X, weights)
    a_final = np.clip(a_values[-1], 1e-10, 1-1e-10)  # Prevent log(0)
    
    # Categorical cross-entropy
    cost = -np.sum(y * np.log(a_final)) / m  
    
    # Add regularization
    if lambda_regulation > 0:
        reg = sum(np.sum(theta[:,1:]**2) for theta in weights)
        cost += (lambda_regulation/(2*m)) * reg
        
    return cost


def backwardPropagation(X, y, weights, lambda_regulation):
    m = X.shape[0]
    a_values, z_values = forwardPropagation(X, weights)
    
    # Output layer delta (softmax derivative)
    deltas = [a_values[-1] - y]  
    
    # Hidden layers (sigmoid derivatives)
    for l in range(len(weights)-2, -1, -1):
        delta = (deltas[-1] @ weights[l+1][:,1:]) * a_values[l+1][:,1:] * (1 - a_values[l+1][:,1:])
        deltas.append(delta)
    
    deltas.reverse()
    
    # Calculate gradients
    gradients = []
    for l in range(len(weights)):
        grad = (deltas[l].T @ a_values[l]) / m
        if lambda_regulation > 0:  # Regularization
            grad[:,1:] += (lambda_regulation/m) * weights[l][:,1:]
        gradients.append(grad)
        
    return gradients


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
    y = np.array(y).reshape(-1, y.shape[1])

    for iteration in range(max_iterations):
        mini_batches = generateMiniBatches(X, y, batch_size)
        total_cost = 0
        accumulated_gradients = [np.zeros_like(w) for w in weights]

        for X_batch, y_batch in mini_batches:
            gradients = backwardPropagation(X_batch, y_batch, weights, lambda_regulation)
            cost = calculateCost(X_batch, y_batch, weights, lambda_regulation)
            total_cost += cost

            for l in range(len(weights)):
                accumulated_gradients[l] += gradients[l]  # Accumulate gradients for the batch

        m = X.shape[0]
        for l in range(len(accumulated_gradients)):
            avg_gradients = accumulated_gradients[l] / len(mini_batches)  # Average gradients over all training examples
            weights[l] -= alpha_learning_rate * avg_gradients

        total_cost /= len(mini_batches)
        print(f"Iteration: {iteration + 1}, Cost: {total_cost:.5f}")

    return weights, accumulated_gradients

def dataNormalization(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    X_range = X_max - X_min
    X_range[X_range == 0] = 1 
    
    X_normalized = 2 * (X - X_min) / X_range - 1 # [-1, +1]

    return X_normalized, X_min, X_max

# The main issue is in the calculateEvaluationMetrics function - it's not properly handling multiclass classification
def calculateEvaluationMetrics(y_true, y_pred):
    # Convert predictions to class indices (0-9 for digits)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    
    # Initialize metrics
    TP = np.zeros(10)  # 10 classes for digits
    FP = np.zeros(10)
    FN = np.zeros(10)
    
    # Calculate per-class metrics
    for cls in range(10):
        TP[cls] = np.sum((y_true_classes == cls) & (y_pred_classes == cls))
        FP[cls] = np.sum((y_true_classes != cls) & (y_pred_classes == cls))
        FN[cls] = np.sum((y_true_classes == cls) & (y_pred_classes != cls))
    
    # Calculate precision, recall, F1 for each class
    precision = np.zeros(10)
    recall = np.zeros(10)
    f1 = np.zeros(10)
    
    for cls in range(10):
        precision[cls] = TP[cls] / (TP[cls] + FP[cls]) if (TP[cls] + FP[cls]) > 0 else 0
        recall[cls] = TP[cls] / (TP[cls] + FN[cls]) if (TP[cls] + FN[cls]) > 0 else 0
        f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls]) if (precision[cls] + recall[cls]) > 0 else 0
    
    # Calculate macro-averaged F1 (average across classes)
    macro_f1 = np.mean(f1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_true_classes)
    
    return accuracy, np.mean(precision), np.mean(recall), macro_f1

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

def evaluateModelAnalysis(k):
    # Load the digits dataset
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    digits_dataset_y = digits[1]

    # Normalize the dataset
    digits_dataset_X, _, _ = dataNormalization(digits_dataset_X)

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    digits_dataset_y = encoder.fit_transform(digits_dataset_y.reshape(-1, 1))

    # Updated configurations
    architectures = [
        [digits_dataset_X.shape[1], 128, 64, 10],
        [digits_dataset_X.shape[1], 64, 10],
        [digits_dataset_X.shape[1], 128, 10]
    ]
    lambda_values = [0.0, 0.0001, 0.001]
    learning_rates = [0.1, 0.05]
    
    results = []

    original_labels = digits[1]
    splits = stratifiedKFoldCrossValidation(digits_dataset_X, original_labels, k=k)

    for arch in architectures:
        for lr in learning_rates:
            for lmbda in lambda_values:
                print(f"\nTesting: {arch}, lambda={lmbda}, alpha={lr}")

                fold_accuracies = []
                fold_f1_scores = []

                for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(splits):
                    print(f"  Fold {fold_idx + 1}/{k}")

                    encoder = OneHotEncoder(sparse_output=False)
                    y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
                    y_test_encoded = encoder.transform(y_test.reshape(-1, 1))
                    
                    weights = generateInitialWeights(arch)

                    trained_weights, _ = trainModel(
                        X_train, y_train_encoded,
                        weights=weights,
                        alpha_learning_rate=lr,
                        lambda_regulation=lmbda,
                        max_iterations=500,  # Increased
                        batch_size=64        # Increased
                    )

                    y_pred = predictClass(X_test, trained_weights)
                    accuracy, precision, recall, f1_score = calculateEvaluationMetrics(y_test_encoded, y_pred)

                    fold_accuracies.append(accuracy)
                    fold_f1_scores.append(f1_score)

                avg_accuracy = np.mean(fold_accuracies)
                avg_f1_score = np.mean(fold_f1_scores)

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

    layer_sizes = [digits_dataset_X.shape[1], 64, 10]
    alpha_learning_rate=0.01
    lambda_regulation=0.1
    max_iterations=300
    batch_size=32
    # epsilon=0.00001

    if case is not None:
        evaluateModelAnalysis(k=10)
    else:       
        # Create 80/20 split for learning curve
        split_ratio = 0.8
        indices = np.random.permutation(digits_dataset_X.shape[0])
        split_idx = int(split_ratio * digits_dataset_X.shape[0])
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        X_train, X_test = digits_dataset_X[train_indices], digits_dataset_X[test_indices]
        y_train, y_test = digits_dataset_y[train_indices], digits_dataset_y[test_indices]
        
        # Train with increasing sample sizes
        training_sizes = []
        test_costs = []
        weights = generateInitialWeights(layer_sizes)
        
        step = max(5, X_train.shape[0] // 50)
        # Ensure y_train_subset is one-hot encoded
        encoder = OneHotEncoder(sparse_output=False)
        y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
        y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

        # Train with increasing sample sizes
        for samples in range(5, X_train.shape[0] + 1, step):
            print(f"Training with {samples}/{X_train.shape[0]} samples.")
            
            X_train_subset = X_train[:samples]
            y_train_subset = y_train_encoded[:samples]  # Use the one-hot encoded labels

            trained_weights, _ = trainModel(
                X_train_subset, y_train_subset,
                weights=weights,
                alpha_learning_rate=alpha_learning_rate,
                lambda_regulation=lambda_regulation,
                max_iterations=max_iterations,
                batch_size=batch_size
            )
            
            test_cost = calculateCost(X_test, y_test_encoded, trained_weights, lambda_regulation)
            training_sizes.append(samples)
            test_costs.append(test_cost)
        
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(training_sizes, test_costs, label="Cost (J)", linewidth=2)
        plt.title(f"Learning Curve - NN: {layer_sizes} - Digits Dataset - Lambda: {lambda_regulation} - Alpha: {alpha_learning_rate}")
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Cost Function (J)")
        plt.legend()
        plt.tight_layout()
        plt.show()