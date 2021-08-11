# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 20:11:44 2021

@author: VARTIKA
"""

# Check the versions of libraries
# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

# Load libraries
import pandas
#from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
names = ['Hours']		#column names 
# Reading data from remote link
#dataset = pandas.read_csv("C:/Users/VARTIKA/Desktop/Student.csv", names=names)
url = "http://bit.ly/w-data"
dataset = pandas.read_csv(url)
# Dimensions of dataset
#shape
print(dataset.shape) 
print(dataset.head(10))
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

dataset.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()
#create a validation dataset
# split-out validation dataset
#array = dataset.values
#X = array[:,0:1]
#Y = array[:,1]
#validation_size = 0.20
#seed = 7
#X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)
X = dataset.iloc[:, :-1].values  
Y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                            test_size=0.2, random_state=0) 



#prediction of logistic regression#
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
#regressor.fit(x.reshape(-1, 1), y)
regressor.fit(X_train.reshape(-1,1), Y_train) 

print("Training complete.")

# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, Y)
plt.plot(X, line);
plt.show()

print(X_test) # Testing data - In Hours
Y_pred = regressor.predict(X_test) # Predicting the score

# Comparing Actual vs Predicted
df = pandas.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
df 

# You can also test with your own data
hours = 9.25
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))

from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(Y_test, Y_pred)) 