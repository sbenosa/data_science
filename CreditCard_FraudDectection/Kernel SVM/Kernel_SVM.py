'''
Author: Sherwin Benosa
Date: April 14,  2019
'''
#+----------------------------------+
#|  CREDIT CARD FRAUD DETECTION     |
#|         KERNEL SVM               |
#+----------------------------------+

# Importing the libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# Importing the dataset
dataset = pd.read_csv('creditcard.csv')

# Drop the Amoun and Time
dataset = dataset.drop(['Amount','Time'],axis=1)

# Separate the X and y values
X = dataset.drop(['Class'],axis=1)
y = dataset['Class']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(random_state = 0, kernel = 'rbf')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Predicting Test Set
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)	
acc = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred)