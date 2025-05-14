from ucimlrepo import fetch_ucirepo
from src.utils import clean_data, binary_target, print_class_distribution


def load_heart_disease_data(binary=True):
    '''Fetch and load UC Irvine data'''
    heart_disease = fetch_ucirepo(id=45)
    X, y = heart_disease.data.features, heart_disease.data.targets

    if binary:
        y = binary_target(y)

    print_class_distribution(y)
    X_clean, y_clean = clean_data(X, y)
    return X_clean, y_clean
