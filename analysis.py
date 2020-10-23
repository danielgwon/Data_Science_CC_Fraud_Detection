import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn


# import the data and pick a sample
data = pd.read_csv("./creditcard.csv")
data = data.sample(frac=0.2, random_state=4)


# histograms
# data.hist(figsize = (20, 20))
# plt.show()


# determine fraud cases
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlier_fraction = len(fraud) / float(len(valid))
print("outlier fraction:", outlier_fraction)
print("Frauds: {}".format(len(fraud)))
print("Valids: {}".format(len(valid)))


# correlation matrix
# corr_mat = data.corr()
# fig = plt.figure(figsize=(12, 9))
# sns.heatmap(corr_mat, vmax=.8, square=True)
# plt.show()


# ORGANIZE THE DATA
# get columns from data frame
columns = data.columns.tolist()

# filter columns to remove data we don't need
columns = [c for c in columns if c not in ['Class']]

# store the variable we're predicting
target = 'Class'

# X includes everything except class column
X = data[columns]
# Y includes all the class labels
Y = data[target]

# print(X.shape)
# print(Y.shape)


# APPLY ALGORITHMS
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# random state
state = 1

# outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,     # number of outliers we think there are
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20,          # make bigger if percentage of outliers is high
                                               contamination=outlier_fraction)
}

# FIT THE MODEL
n_outliers = len(fraud)
for i, (clf_name, clf) in enumerate(classifiers.items()):

    # fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

    # reshape the prediction so valid = 0 and fraud = 1
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    # calculate number of errors
    n_errors = (y_pred != Y).sum()

    # classification matrix
    print("{}: {}".format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))
