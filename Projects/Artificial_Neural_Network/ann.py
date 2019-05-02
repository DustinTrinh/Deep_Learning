# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#Reduce overfitting by using Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#open the dataset and determine which variable is more likely to impact the study. (Not CustomerID, nor name)
#It should be CreditScore/Geography/Gender/Age...
#SO the index should be from 3 to 12
#The : is excluded so it will be 3:13
X = dataset.iloc[:, 3:13].values
#y is the last column, exited or not, so the index is 13
y = dataset.iloc[:, 13].values

# Encoding categorical data
#So when we see X array, there are 2 independent variables : Country and Gender. 
#So we perform encode to Country first
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#First : is all the line of the matrix, and then from column 2 till end
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part 2 - Lets Make an ANN

#initializing ANN
classifier = Sequential()

#adding the input layer and the first hidden layer
#Number of nodes should be 11 since the input node is 11 (we have 11 Independent Variable)
#and the output layer is 1. So (11+1)/2  = 6
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu', input_dim=11))
classifier.add(Dropout(rate = 0.1))

#Adding second layer
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

#Adding output layer
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting ANN to training set
classifier.fit(X_train, y_train, batch_size = 20, epochs = 100)

#Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#return boolean
y_pred = (y_pred > 0.5) 


#Test
'''
Geography: France
Credit score : 600
Gender : Male
Age: 40
Tenture : 3
Balance : 60000
Number of products: 2
Has A credit card : Yes
Is Active member : Yes
Estimate Salary : 50000
'''
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5) 

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Part 4 - Evaluating, Improving and Tuning the ANN

#Evaluating ANN
#Testing using K-FOld
#Mean is to know the general accuracy
#Variance is to see if it spike the chance. The lower the better

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu', input_dim=11))
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()

#Improving the ANN
#Dropout Regularization to reduce overfitting if needed line 76 and line 80

#Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu', input_dim=11))
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size' : [25, 32], 
              'epochs' : [100,500],
              'optimizer' : ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy =grid_search.best_score_
