import numpy as np
from GNBwSWClassifier import GaussianNaiveBayesWithSlidingWindow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Define initial parameters
window_size = 20
var_smoothing = 1e-9

# Create a Gaussian Naive Bayes classifier with sliding window
gnb = GaussianNaiveBayesWithSlidingWindow(window_size=window_size, var_smoothing=var_smoothing)

# Define number of samples and features
n_samples = 100
n_features = 3

# Generate initial dataset
X = np.random.rand(n_samples, n_features) * 20 - 10
y = []
# Generate initial labels
for x in X:
    y.append(int(x[1] > 0))

# Train the classifier with the initial dataset
for i in range(n_samples):
    gnb.learn_one(X[i], y[i])

# Calculate current mean of feature 1
sum_feature_1 = np.sum(X[:][1])
mean_feature_1 = sum_feature_1 /n_samples
print("Mean of feature 1 : " + str(mean_feature_1))

pred_arr = []
truth_y = []
# Gradually change the mean of feature 1 over time
for i in range(n_samples, n_samples * 2):
    # Generate new data point from range (-10,10)
    X_new = np.random.rand(n_features) * 20 - 10
    # Gradually changing mean of feature 1
    X_new += np.array([0, i/n_samples * 6.9, 0])
    # Assign label based on a threshold of feature 1
    y_new = int(X_new[1] > 5)
    # Train the classifier with the new data point
    gnb.learn_one(X_new, y_new)
    # Generate test data point
    X_test = np.random.rand(n_features)  * 20 - 10 + np.array([0, i/n_samples * 6.9, 0])
    y_test = int(X_test[1] > 5)

    # Calculate current mean of feature 1
    sum_feature_1 += X_new[1]
    mean_feature_1 = sum_feature_1 / i
    print("\n Mean of feature 1 : " + str(mean_feature_1))

    # Predict the label of a test data point
    y_pred = gnb.predict_one(X_test)
    pred_arr.append(y_pred)
    truth_y.append(y_test)
    print(f"{(i-n_samples)} True label: {y_new}, Predicted label: {y_pred}")

accuracy = accuracy_score(truth_y, pred_arr)
precision = precision_score(truth_y, pred_arr)
recall = recall_score(truth_y, pred_arr)
f1 = f1_score(truth_y, pred_arr)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 score: {f1:.2f}")