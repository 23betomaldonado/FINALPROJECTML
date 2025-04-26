import numpy as np
from collections import Counter

def partition(x):
    """Partition the column vector x into subsets indexed by its unique values"""
    unique_values = np.unique(x)
    partitions = {}
    for value in unique_values:
        partitions[value] = np.where(x == value)[0]
    return partitions

def entropy(y):
    """Compute the entropy of vector y"""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def mutual_information(x, y):
    """Compute mutual information between feature x and labels y"""
    entropy_y = entropy(y)
    unique_values, value_counts = np.unique(x, return_counts=True)
    conditional_entropy = 0
    for value, count in zip(unique_values, value_counts):
        indices = np.where(x == value)[0]
        subset_entropy = entropy(y[indices])
        conditional_entropy += (count / len(x)) * subset_entropy
    return entropy_y - conditional_entropy

def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """ID3 algorithm to build decision tree"""
    if attribute_value_pairs is None:
        attribute_value_pairs = []
        for attr_idx in range(x.shape[1]):
            unique_values = np.unique(x[:, attr_idx])
            for value in unique_values:
                attribute_value_pairs.append((attr_idx, value))
    
    # Termination conditions
    if len(np.unique(y)) == 1:
        return y[0]
    if not attribute_value_pairs or depth >= max_depth:
        return Counter(y).most_common(1)[0][0]
    
    # Find best split
    best_gain, best_pair = -1, None
    for attr_idx, value in attribute_value_pairs:
        gain = mutual_information(x[:, attr_idx] == value, y)
        if gain > best_gain:
            best_gain, best_pair = gain, (attr_idx, value)
    
    if not best_pair:
        return Counter(y).most_common(1)[0][0]
    
    attr_idx, value = best_pair
    split_indices = x[:, attr_idx] == value
    remaining_pairs = [p for p in attribute_value_pairs if p != (attr_idx, value)]
    
    # Build subtrees
    tree = {
        (attr_idx, value, True): id3(x[split_indices], y[split_indices], remaining_pairs, depth+1, max_depth),
        (attr_idx, value, False): id3(x[~split_indices], y[~split_indices], remaining_pairs, depth+1, max_depth)
    }
    return tree

def predict_example(x, tree):
    """Predict label for single example using decision tree"""
    if not isinstance(tree, dict):
        return tree
    for (attr_idx, value, flag), subtree in tree.items():
        if (x[attr_idx] == value) == flag:
            return predict_example(x, subtree)
    return list(tree.values())[0]

def compute_error(y_true, y_pred):
    """Compute classification error"""
    return np.mean(y_true != y_pred)

def visualize(tree, depth=0, parent_feature=None):
    """Visualize decision tree with better formatting"""
    if depth == 0:
        print("DECISION TREE")
        print("=" * 50)
        print("Feature Mapping:")
        features = [
            "Q1:Gender", "Q2:CollegeYear", "Q3:Major", "Q4:GPA", 
            "Q5:TakenOnline", "Q6:TechComfort", "Q7:LearningStyle",
            "Q8:InternetQuality", "Q9:Race", "Q10:WillingnessOnline"
        ]
        for i, feat in enumerate(features):
            print(f"x{i} = {feat}")
        print("\n" + "=" * 50 + "\n")
    
    for (attr_idx, value, flag), subtree in tree.items():
        feature_name = f"x{attr_idx}"
        condition = f"{feature_name} == {value}" if flag else f"{feature_name} != {value}"
        
        # Print the current node
        print('    ' * depth + f"├── {condition}")
        
        # Print the subtree or leaf
        if isinstance(subtree, dict):
            visualize(subtree, depth+1, feature_name)
        else:
            print('    ' * (depth+1) + f"└── Predict: {subtree}")
def main():
    # Load your test data (we'll split it into train/test)
    test_data = np.array([
        [0,3,0,1,1,1,0,1,0,1,0], [0,3,0,1,1,2,1,1,0,1,0],
        [0,3,0,0,1,2,2,1,0,2,0], [0,3,2,2,1,1,2,1,1,1,0],
        [0,1,2,3,1,1,2,1,5,1,0], [0,3,0,2,0,1,2,1,5,2,0],
        [0,1,0,3,1,1,1,1,1,1,1], [1,3,0,1,1,1,0,1,1,1,0],
        [0,2,0,2,1,1,1,1,3,1,1], [0,3,0,3,0,1,1,1,1,0,1],
        [1,1,0,3,1,2,1,1,1,1,1], [1,2,0,1,1,1,2,1,3,1,1],
        [0,3,0,2,1,2,2,1,1,1,0], [0,3,0,3,1,1,2,1,3,1,1],
        [0,3,0,2,1,1,2,1,2,1,0], [1,3,0,1,1,1,1,1,3,2,1],
        [0,2,0,2,0,2,2,1,4,2,1], [0,3,0,3,1,1,1,1,5,1,1],
        [0,3,0,3,1,1,2,1,1,2,1], [0,0,0,3,0,1,2,1,4,1,1],
        [1,3,0,3,1,1,2,1,3,1,1], [0,3,0,2,1,1,1,1,4,1,0],
        [1,3,0,3,1,2,2,1,4,1,0], [0,3,0,2,1,1,2,1,1,1,1],
        [0,2,0,3,1,2,2,1,1,1,0], [0,3,0,3,1,1,2,1,0,1,0],
        [1,2,0,2,1,1,1,0,1,1,0], [1,2,0,2,1,1,1,1,1,1,0],
        [0,3,0,2,1,1,2,1,5,1,0], [0,3,0,3,1,1,1,1,1,1,0],
        [0,3,0,3,1,1,1,1,0,1,1], [0,3,0,1,1,1,0,1,0,1,1],   
        [0,3,0,1,1,2,1,1,0,1,1], [0,3,0,0,1,2,2,1,0,2,0],    
        [0,3,1,2,1,1,2,1,1,1,0], [0,1,2,3,1,1,2,1,5,1,0],    
        [0,3,0,2,0,1,2,1,5,2,0], [0,1,0,3,1,1,1,1,1,1,1],    
        [1,3,0,1,1,1,0,1,1,1,0], [0,2,0,2,1,1,1,1,4,1,1],    
        [0,3,0,3,0,1,1,1,1,0,1], [1,1,0,3,1,2,1,1,1,1,1],    
        [1,2,0,1,1,1,2,1,3,1,1], [0,3,0,2,1,2,2,1,1,1,0],    
        [0,3,0,3,1,1,2,1,1,1,0], [0,3,0,2,1,1,2,1,2,1,0],   
        [1,3,0,1,1,1,1,1,3,2,1], [0,2,0,2,0,2,2,1,4,2,1],    
        [0,3,0,3,1,1,1,1,5,1,1], [0,3,0,3,1,1,2,1,1,2,1], 
        [0,0,0,3,0,1,2,1,4,1,1], [0,3,0,1,1,2,1,1,0,1,0]     
    ])
    
    # Split into features (Q1-Q10) and target (Q11)
    X = test_data[:, :-1]
    y = test_data[:, -1]
    
    # Split into train (80%) and test (20%)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, y_train = X[indices[:split]], y[indices[:split]]
    X_test, y_test = X[indices[split:]], y[indices[split:]]
    
    # Train decision tree
    print("Training decision tree...")
    tree = id3(X_train, y_train, max_depth=3)
    
    # Visualize and evaluate
    visualize(tree)
    y_pred = [predict_example(x, tree) for x in X_test]
    error = compute_error(y_test, y_pred)
    print(f"\nTest Error: {error*100:.2f}%")
    
    # Show feature mapping
    print("\nFeature Mapping:")
    features = [
        "Q1:Gender", "Q2:CollegeYear", "Q3:Major", "Q4:GPA", 
        "Q5:TakenOnline", "Q6:TechComfort", "Q7:LearningStyle",
        "Q8:InternetQuality", "Q9:Race", "Q10:WillingnessOnline"
    ]
    for i, feat in enumerate(features):
        print(f"x{i} = {feat}")

if __name__ == "__main__":
    main()