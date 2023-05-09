from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from GNBwSWClassifier import GaussianNaiveBayesWithSlidingWindow
import numpy as np

# load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# split the data into test and train sets using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

# create the GaussianNaiveBayesWithSlidingWindow model
nb = GaussianNaiveBayesWithSlidingWindow()

# train the model
for xi, yi in zip(X_train, y_train):
    nb.learn_one(xi, yi)


# let the model predict the labels of randomly generated float data points
pred_arr = []
truth_y = []
for xi, yi in zip(X_test, y_test):
    pred = nb.predict_one(xi)
    pred_arr.append(pred)
    truth_y.append(yi)
    

print(pred_arr.count(1))
print(pred_arr.count(0))

# compute the metrics and print them
accuracy = accuracy_score(truth_y, pred_arr)
precision = precision_score(truth_y, pred_arr)
recall = recall_score(truth_y, pred_arr)
f1 = f1_score(truth_y, pred_arr)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 score: {f1:.2f}")

