'''
Author: Sherwin Benosa
Date: March 17,  2018
'''
#+----------------------------------+
#|  SUPPORT VECTOR MACHINE (SVM)    |
#+----------------------------------+

# Importing the libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

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

# Fitting SVM (RBF) to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)

# Predicting Test Set
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

