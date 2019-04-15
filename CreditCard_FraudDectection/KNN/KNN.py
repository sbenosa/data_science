'''
Author: Sherwin Benosa
Date: April 14,  2019
'''
#+----------------------------------+
#|  CREDIT CARD FRAUD DETECTION     |
#|     K NEAREST NEIGHBORS          |	
#+----------------------------------+

# Importing the libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# Importing the dataset
dataset = pd.read_csv('credit_card.csv',index_col = 0)

# Drop the Amoun and Time
dataset = dataset.drop(['Amount','Time'],axis=1)

# Separate the X and y values
X = dataset.drop(['Class'],axis=1)
y = dataset['Class']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
