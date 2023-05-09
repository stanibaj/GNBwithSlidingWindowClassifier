from GNBwSWClassifier import GaussianNaiveBayesWithSlidingWindow
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


# # Create two classifiers with different var_smoothing values
# clf1 = GaussianNaiveBayesWithSlidingWindow(var_smoothing=1e-9)
# clf2 = GaussianNaiveBayesWithSlidingWindow(var_smoothing=100) # tenhle dosahuje lepsich vysledku

# # Train the classifiers with some data
# X_train = np.random.rand(200, 3)
# y_train = np.random.randint(0, 2, size=200)
# for i in range(len(X_train)):
#     clf1.learn_one(X_train[i], y_train[i])
#     clf2.learn_one(X_train[i], y_train[i])

# # Test the classifiers on some data
# X_test = np.random.rand(10, 3)
# y_test = np.random.randint(0, 2, size=10)

# clf1_pred = []
# clf2_pred = []
# for i in range(len(X_test)):
#     pred1 = clf1.predict_one(X_test[i])
#     pred2 = clf2.predict_one(X_test[i])
#     clf1_pred.append(pred1)
#     clf2_pred.append(pred2)
#     print(f"True label: {y_test[i]}, var_smoothing=1e-9 prediction: {pred1}, var_smoothing=1e-7 prediction: {pred2}")

# print("\nCLF 1")
# accuracy = accuracy_score(y_test, clf1_pred)
# precision = precision_score(y_test, clf1_pred)
# recall = recall_score(y_test, clf1_pred)
# f1 = f1_score(y_test, clf1_pred)
# print(f"Accuracy: {accuracy:.2f} Precision: {precision:.2f} Recall: {recall:.2f} F1 score: {f1:.2f}")

# print("\nCLF 2")
# accuracy = accuracy_score(y_test, clf2_pred)
# precision = precision_score(y_test, clf2_pred)
# recall = recall_score(y_test, clf2_pred)
# f1 = f1_score(y_test, clf2_pred)
# print(f"Accuracy: {accuracy:.2f} Precision: {precision:.2f} Recall: {recall:.2f} F1 score: {f1:.2f}")

from GNBwSWClassifier import GaussianNaiveBayesWithSlidingWindow
import numpy as np

def test_numerical_stability():
    # Generate a dataset with a 3 features.
    X = np.random.rand(100, 3)

    # Set the value of feature 0 to be constant to simulate zero variance.
    X[:, 0] = np.mean(X[:, 0])
    
    # Set the first half of the data points to be label 1.
    X[:50, 1] = np.random.uniform(3.1,10.0,size=(50,))
    X[:50, 2] = np.random.uniform(-5.0,1.9,size=(50,))

    # Set the second half of the data points to be random from range [-10.0,10> .
    X[50:, 1] = np.random.uniform(-10.0,10.0,size=(50,))
    X[50:, 2] = np.random.uniform(-10.0,10.0,size=(50,))

    # Shuffle the data points.
    np.random.shuffle(X)

    # Label the data points:
    #   if feature 1 is greater than 3 and feature 2 is lower than 2 => label is 1
    #   else label is 0
    y = []
    for i in range(len(X)):
        y_val = 1 if X[i][1] > 3 and X[i][2] < 2 else 0
        y.append(y_val)

    # Split the data into test and train sets using train_test_split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
    print(y_train.count(1))
    print(y_train.count(0))

    # Initialize the classifier with window size 10 and default value of var_smoothing parameter.
    clf = GaussianNaiveBayesWithSlidingWindow(window_size=10)

    # Train the classifier with the dataset.
    for i in range(len(X_train)):
        clf.learn_one(X_train[i], y_train[i])

    # Let the model predict the labels.
    pred_arr = []
    truth_y = []
    for xi, yi in zip(X_test, y_test):
        pred = clf.predict_one(xi)
        pred_arr.append(pred)
        truth_y.append(yi)

    print(pred_arr.count(1))
    print(pred_arr.count(0))

    # Make a prediction using a test data point with zero variance feature
    # x_test = np.array([0,1,2]) # => should have label 0
    # y_pred = clf.predict_one(x_test)

    # # Check if the predicted class is correct (should be random)
    # assert y_pred in [0, 1], "Prediction should be 0 or 1"
    # print("X")
    # print(str(X))
    # print(str(x_test) + " > " + str(y_pred))
    

    # Compute the metrics and print them.
    accuracy = accuracy_score(truth_y, pred_arr)
    precision = precision_score(truth_y, pred_arr)
    recall = recall_score(truth_y, pred_arr)
    f1 = f1_score(truth_y, pred_arr)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")

# if __name__ == "__main__":
test_numerical_stability()
