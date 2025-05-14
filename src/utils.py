import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def clean_data(X, y):
    """ Drop rows with any missing values in either X or y """
    df = pd.concat([X, y], axis=1)
    df.dropna(inplace=True)
    y_clean = df[y.columns[0]]  
    X_clean = df.drop(columns=[y.columns[0]])
    return X_clean, y_clean

def binary_target(y):
    '''Convert multi class 1-4 n to binary 0 - 1'''
    y_binary = y.copy()
    y_binary[y_binary > 0] = 1
    return y_binary

def print_class_distribution(y):
    """Print distribution of binary classes."""
    print("Target distribution:")
    print(y.value_counts())

def save_report(report_str, path):
    '''Save to text file'''
    with open(path, 'w') as f:
        f.write(report_str)