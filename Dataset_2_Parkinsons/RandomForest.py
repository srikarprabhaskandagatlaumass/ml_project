import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

loan_data=pd.read_csv("Dataset_2_Parkinsons/parkinsons.csv")


class DecisionTreeNode:
    def __init__(self, name, depth=0, split_val=None):
        self.name =name
        self.children ={}
        self.depth =depth
        self.split_value = split_val
        self.is_numerical =False

from collections import Counter
def entropy(x):
    if len(x) ==0:
        return 0

    counts=Counter(x)
    entropy=0
    total=len(x)

    for count in counts.values():
        prob =count/total
        if prob >0:
            entropy +=-prob * np.log2(prob)

    return entropy

def best_nm_split(x_col,y_labels):

    if len(np.unique(x_col)) < 2:
        return None, float('-inf')

    threshold =np.median(x_col)

    #Calculate information gain
    total_entropy =entropy(y_labels)

    left_mask = x_col <=threshold
    right_mask = x_col >threshold

    if np.any(left_mask) and np.any(right_mask):
        left_entryp =entropy(y_labels[left_mask])
        right_entryp=entropy(y_labels[right_mask])

        left_weight =np.sum(left_mask) /len(y_labels)
        right_weight=np.sum(right_mask) /len(y_labels)

        gain=total_entropy -(left_weight * left_entryp +right_weight *right_entryp)
        return threshold, gain

    return None, float('-inf')

import random
def best_attribute(x_train,y_train, L, m_features=None):
    if m_features is not None and m_features<len(L):
        L = random.sample(L, m_features)

    total_entropy=entropy(y_train)
    best_gain = float('-inf')
    best_attr = None
    best_split_value = None
    is_numerical = False

    for A in L:
        if pd.api.types.is_numeric_dtype(x_train[A]):
            # Handle numerical attribute
            threshold, gain = best_nm_split(x_train[A], y_train)

            if gain >best_gain:
                best_gain = gain
                best_attr = A
                best_split_value = threshold
                is_numerical = True
        else:
            unique_vals = np.unique(x_train[A])
            weighted_entropy = 0

            for val in unique_vals:
                subset_y =y_train[x_train[A] == val]
                entropy_val =entropy(subset_y)
                subset_size = len(subset_y)
                prob_val = subset_size / len(y_train)
                weighted_entropy +=prob_val * entropy_val

            gain=total_entropy-weighted_entropy

            if gain>best_gain:
                best_gain =gain
                best_attr =A
                best_split_value =None
                is_numerical =False

    return best_attr, best_split_value,is_numerical

def Decision_Tree(x_train, y_train,L,max_depth=None,m_features=None,current_depth=0):

    unique_labels =np.unique(y_train)
    majority_label =y_train.mode()[0]

    if len(unique_labels) ==1:
        return DecisionTreeNode(name=unique_labels[0], depth=current_depth)

    if max_depth is not None and current_depth >= max_depth:
        return DecisionTreeNode(name=majority_label, depth=current_depth)

    if len(L) ==0:
        return DecisionTreeNode(name=majority_label, depth=current_depth)

    # Find best attribute to split on
    best_A,split_value, is_numerical = best_attribute(x_train, y_train, L,m_features)

    if best_A is None:
        return DecisionTreeNode(name=majority_label,depth=current_depth)

    # Create node
    node = DecisionTreeNode(name=best_A, depth=current_depth, split_val=split_value)
    node.is_numerical =is_numerical

    if is_numerical:
        left_mask =x_train[best_A] <= split_value
        right_mask =x_train[best_A] >split_value


        node.children['left'] =Decision_Tree(
            x_train[left_mask], y_train[left_mask], L,
            max_depth, m_features, current_depth + 1
        ) if sum(left_mask) > 0 else DecisionTreeNode(name=majority_label,depth=current_depth+1)


        node.children['right']=Decision_Tree(
            x_train[right_mask], y_train[right_mask], L,
            max_depth, m_features, current_depth + 1
        ) if sum(right_mask) > 0 else DecisionTreeNode(name=majority_label,depth=current_depth+1)
    else:
        # Categorical split
        unique_vals=np.unique(x_train[best_A])

        for val in unique_vals:
            subset_mask =x_train[best_A] == val
            node.children[val] =Decision_Tree(
                x_train[subset_mask], y_train[subset_mask], L,
                max_depth, m_features,current_depth + 1
            ) if sum(subset_mask) > 0 else DecisionTreeNode(name=majority_label, depth=current_depth+1)

    return node

def DecisionTreeClassifier(x_train, y_train,x_test,max_depth=5,m_features=None):
    L=list(x_train.columns)
    majority_class =y_train.mode()[0]
    root =Decision_Tree(x_train, y_train, L,max_depth, m_features)

    pred_list =[]
    class_labels =np.unique(y_train)

    for i in range(len(x_test)):
        row =x_test.iloc[i]
        node =root

        while True:
            if node.name in class_labels:
                pred_list.append(node.name)
                break

            if node.is_numerical:

                direction ='left' if row[node.name] <= node.split_value else 'right'
                node=node.children.get(direction,
                       DecisionTreeNode(name=majority_class))
            else:

                val=row[node.name]
                node=node.children.get(val,
                       DecisionTreeNode(name=majority_class))

    return pred_list

def create_bootstrap(x_train, y_train):
    n_samples =len(x_train)
    indices =np.random.choice(n_samples, size=n_samples,replace=True)
    return x_train.iloc[indices],y_train.iloc[indices]

class RandomForestClassifier:
    def __init__(self, ntree=10, max_depth=None, m_features='sqrt'):
        self.ntree = ntree
        self.max_depth =max_depth
        self.m_features=m_features
        self.trees =[]
        self.classes_ =None

    def fit(self,x_train,y_train):
        self.trees =[]
        self.classes_=np.unique(y_train)

        if self.m_features =='sqrt':
            m =int(np.sqrt(len(x_train.columns)))
        elif isinstance(self.m_features, float):
            m =int(self.m_features *len(x_train.columns))
        else:
            m=self.m_features

        for i in range(self.ntree):
            x_boot, y_boot =create_bootstrap(x_train, y_train)

            tree = Decision_Tree(
                x_boot,
                y_boot,
                list(x_train.columns),
                max_depth=self.max_depth,
                m_features=m
            )
            self.trees.append(tree)

    def predict(self, x_test):
        all_preds =[]

        for tree in self.trees:
            preds =[]

            for i in range(len(x_test)):
                row =x_test.iloc[i]
                node =tree

                while True:
                    if node.name in self.classes_:
                        preds.append(node.name)
                        break

                    if node.is_numerical:
                        direction='left' if row[node.name] <= node.split_value else 'right'
                        if direction in node.children:
                            node=node.children[direction]
                        else:
                            preds.append(self.classes_[0])
                            break
                    else:
                        # Categorical split
                        val=row[node.name]
                        if val in node.children:
                            node =node.children[val]
                        else:
                            preds.append(self.classes_[0])
                            break
            all_preds.append(preds)

        # Majority voting
        final_preds =[]
        for sample_preds in zip(*all_preds):
            final_preds.append(Counter(sample_preds).most_common(1)[0][0])

        return final_preds
    
def stratified_kfold(x, y, k=5):

    classes = np.unique(y)
    class_indices = {cls:np.where(y ==cls)[0] for cls in classes}

    # Shuffle the indices
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])

    # initializztion of folds
    folds = [[] for i in range(k)]

    # Distribute of each calss
    for cls in classes:
        cls_indices =class_indices[cls]
        fold_sizes =[len(cls_indices) // k] *k
        for i in range(len(cls_indices) % k):
            fold_sizes[i] += 1

        current = 0
        for i in range(k):
            start, end = current, current + fold_sizes[i]
            folds[i].extend(cls_indices[start:end])
            current = end

    # Generate train-test splits
    for i in range(k):
        test_indices =folds[i]
        train_indices=[idx for j, fold in enumerate(folds) if j != i for idx in fold]
        yield train_indices,test_indices

def accuracy(y_true,y_pred):

    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def precision(y_true, y_pred,pos_label):

    true_positives = sum(1 for true, pred in zip(y_true, y_pred)
                       if true == pos_label and pred == pos_label)
    predicted_positives = sum(1 for pred in y_pred if pred == pos_label)

    return true_positives / predicted_positives if predicted_positives > 0 else 0

def recall(y_true,y_pred,pos_label):

    true_positives = sum(1 for true, pred in zip(y_true, y_pred)
                   if true == pos_label and pred == pos_label)
    actual_positives = sum(1 for true in y_true if true ==pos_label)

    return true_positives /actual_positives if actual_positives > 0 else 0

def f1(y_true, y_pred, pos_label):

    prec=precision(y_true, y_pred, pos_label)
    rec= recall(y_true, y_pred, pos_label)

    if (prec + rec) ==0:
        return 0
    return 2 * (prec* rec)/(prec + rec)

def multicalss_metrics(y_true, y_pred):
    classes = np.unique(y_true)
    metrics ={
        'precision': 0,
        'recall': 0,
        'f1': 0
    }
    total_samples=len(y_true)

    for cls in classes:
        # Calculate metrics for each class
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred != cls)

        # Precision for this class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Recall for this class
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1 for this class
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Weight by class prevalence
        class_weight = sum(1 for true in y_true if true == cls) / total_samples

        metrics['precision'] += precision * class_weight
        metrics['recall'] += recall * class_weight
        metrics['f1'] += f1 * class_weight

    return metrics

def evaluate_random_forest(x,y,ntree_values,k=5,max_depth=None):
    results = {
        'ntree': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    # Determine if binary or multiclass
    classes = np.unique(y)
    is_binary =len(classes)== 2

    for ntree in ntree_values:
        print(f"\nEvaluating ntree ={ntree}")

        fold_metrics = []

        for train_idx, test_idx in stratified_kfold(x, y, k):
            x_train, x_test =x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test =y.iloc[train_idx], y.iloc[test_idx]

            rf = RandomForestClassifier(ntree=ntree, max_depth=max_depth)
            rf.fit(x_train,y_train)
            preds = rf.predict(x_test)

            # Calculate metrics
            acc=accuracy(y_test,preds)

            if is_binary:
                pos_label = classes[1]  # Assuming second class is positive
                prec= precision(y_test, preds, pos_label)
                rec=recall(y_test, preds, pos_label)
                f1score= f1(y_test, preds, pos_label)
            else:
                metrics = multicalss_metrics(y_test, preds)
                prec = metrics['precision']
                rec = metrics['recall']
                f1score= metrics['f1']

            fold_metrics.append((acc, prec, rec, f1score))

        # Store average results
        avg_metrics = np.mean(fold_metrics, axis=0)
        results['ntree'].append(ntree)
        results['accuracy'].append(avg_metrics[0])
        results['precision'].append(avg_metrics[1])
        results['recall'].append(avg_metrics[2])
        results['f1'].append(avg_metrics[3])

    return pd.DataFrame(results)

def plot_metrics(results_df, dataset_name):
    """
    Create 4 metric plots (accuracy, precision, recall, f1) for a given dataset
    """
    metrics =['accuracy', 'precision', 'recall', 'f1']
    ntree_values =results_df['ntree']

    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        plt.plot(ntree_values, results_df[metric], marker='o', linestyle='-')
        plt.title(f'{metric.capitalize()} vs Number of Trees',fontsize=12)
        plt.xlabel('Number of Trees (ntree)')
        plt.ylabel(metric.capitalize())
        plt.grid(True)

        # Add value labels
        for x, y in zip(ntree_values, results_df[metric]):
            plt.text(x, y, f'{y:.3f}', ha='center',va='bottom')

    plt.tight_layout()
    plt.suptitle(f'Random Forest Performance Metrics - {dataset_name} Dataset', y=1.02, fontsize=14)
    plt.show()


if __name__ == "__main__":
    results = evaluate_random_forest(
     x=loan_data.drop('Diagnosis', axis=1),
     y=loan_data['Diagnosis'],
     ntree_values=[1, 5,10,20,30,40,50],
     k=5,
     max_depth=10
)
    print("\n" + "="*80)
    print("Wisconsin Breast Cancer Dataset Performance Graphs")
    print("="*80)
    plot_metrics(results, "Wisconsin Breast Cancer")
    print(results)