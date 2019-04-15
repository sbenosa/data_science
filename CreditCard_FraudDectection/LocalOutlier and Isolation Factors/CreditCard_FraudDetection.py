'''
Author: Sherwin Benosa
Date: April 13,  2019
'''
#+----------------------------------+
#|  CREDIT CARD FRAUD DETECTION     |
#|   ISOLATION FOREST ALGORITHM     |	
#+----------------------------------+

# Importing the libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# Importing the dataset
dataset = pd.read_csv('creditcard.csv',index_col=0)

# Reduce the dataset to 10% inorder to conserve resources
dataReduce = dataset.sample(frac=0.1, random_state = 1)

# Separate the X and y values
X = dataReduce.drop(['Class'],axis=1)
y = dataReduce['Class']

# Determine the outlier fraction
fraud = dataReduce[dataReduce['Class'] == 1]
valid = dataReduce[dataReduce['Class'] == 0]

outlier_fraction = len(fraud)/float(len(valid))


# Importing the sklear libraries
from sklearn.metrics import classification_report, accuracy_score  
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# define random states
state = 1

# define outlier detection tools to be compared
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=outlier_fraction)}

# Fit the model
plt.figure(figsize=(9, 7))
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
    
	# Reshape the prediction values to 0 for valid, 1 for fraud. 
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    #Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))