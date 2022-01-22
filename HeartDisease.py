# The code reads data of patients across different parameters
# and makes a prediction about heart disease risk based on
# data provided.


from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
import os
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.svm import SVC
import matplotlib
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# opens the training and validation dataset
data = pd.read_csv('Correct_Dataset.csv')

# opens the testing dataset
test = pd.read_csv('heart_data.csv')

# gets all column headings from training
col = data.columns.values.tolist()

# gets all column headings from test
categories = [column for column in test]

# checks if test file has all key data
if 'Exercised_Induced_Angina' not in categories or 'Colestrol' not in categories or 'Rest_ECG' not in categories:
    print('Accurate prediction cannot be made')
# if key data available, following code is run
# find the non-key attributes missing in test file
else:
    missing_factors = []
    for j in col:
        if j not in categories:
            missing_factors.append(j)

    # remove rows that have a missing value

    for i in col:
        data.drop(data[data[i] == '?'].index, inplace = True)

    data.head()
    data.isnull().sum()
    # loads the target data into y from training
    y = data['Target']
    # Listing columns that need to be dropped
    data_drop = ['Serial','Target']
    data_drop.extend(missing_factors)
    # loading the data into X without columns that have to be dropped
    X = data.drop(data_drop, axis = 1)
    # splitting the data into train and validation with an 80-20 split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20)
    X.head()
    X.shape
    clf = SVC(gamma = 'auto')
    # training and validation
    # fitting model on x_train to make predictions on x_val
    clf.fit(X_train, y_train)
    print (f'Training Accuracy  : {clf.score(X_train,y_train):.3f}')
    preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, preds)
    print (f'Validation Accuracy : {clf.score(X_val,y_val):.3f}')
    #### Histogram plot
    #data.hist()
    #pyplot.show()

#### testing new dataset
    # dropping missing value rows from test file
    for i in categories:
        test.drop(test[test[i] == '?'].index, inplace = True)

    test.head()
    test.isnull().sum()

# loads data into variables

    X.head()
    y_test = test['Target']
    X_test = test.drop(['Serial', 'Target'], axis=1)

# prediction in line with fitted model

    test_preds = clf.predict(X_test)

    print (f'Prediction Accuracy  : {clf.score(X_test,y_test):.3f}')


