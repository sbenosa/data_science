'''
Author: Sherwin Benosa
Date: March 17,  2018
'''
# Assumption: This is the best model to classify if  cancer cell is malignant or benign 
# Since this is already selected over the other models, we need to do paramenter tuning to further
# enhance its perfomance and make it more robust in the end

#+----------------------------------+
#|   RANDON FOREST CLASSIFICATION   |
#+----------------------------------+

# Importing the libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import time

# Importing the dataset
dataset = pd.read_csv('data_breast_cancer.csv',index_col=0)
X = dataset.drop(['target'],axis=1)
y = dataset['target']
	
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### Model Building #####
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,
                                    criterion = 'entropy')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest (n=100)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

## K-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X= X_train, y = y_train,
                             cv = 10)
print("Random Forest Classifier Accuracy: %0.2f (+/- %0.2f)"  % (accuracies.mean(), accuracies.std() * 2))

#+################################+
#|       PARAMETER TUNING         |
#+################################+

print('#|       PARAMETER TUNING         |')
# print('Round 1: Entropy')
# # TUNING Entropy
# parameters = {"max_depth": [3, None],
#               "max_features": [1, 5, 10],
#               'min_samples_split': [2, 5, 10],
#               'min_samples_leaf': [1, 5, 10],
#               "bootstrap": [True, False],
#               "criterion": ["entropy"]}

# from sklearn.model_selection import GridSearchCV
# grid_search = GridSearchCV(estimator = classifier, # Make sure classifier points to the RF model
#                            param_grid = parameters,
#                            scoring = "accuracy",
#                            cv = 10)
#                            #n_jobs = 1)  # this parameter depending on the value will take resource available

# t0 = time.time()
# grid_search = grid_search.fit(X_train, y_train)
# t1 = time.time()
# print("Took %0.2f seconds" % (t1 - t0))

# rf_best_accuracy = grid_search.best_score_
# rf_best_parameters = grid_search.best_params_
# rf_best_accuracy, rf_best_parameters

print('Round 2: Entropy')
parameters = {"max_depth": [None],
              "max_features": [3, 5, 7],
              'min_samples_split': [8, 10, 12],
              'min_samples_leaf': [1, 2, 3],
              "bootstrap": [True],
              "criterion": ["entropy"]}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier, # Make sure classifier points to the RF model
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10)
                           #n_jobs = 1)  # this parameter depending on the value will take resource available

t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters


# Predicting Test Set
y_pred = grid_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest (n=100, GSx2 + Entropy)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



