import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filename, size):
    ## load data
    custom_columns = ['feature 1', 'feature 2', 'label']
    data_file = pd.read_csv(filename, header=None, names=custom_columns)
    data_file = data_file.dropna()
    
    # make train and test sets
    data_x = np.array(data_file.drop('label', axis=1))
    data_y = np.array(data_file['label'])
    
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,
                                                        test_size=size,
                                                        stratify=data_y,
                                                        random_state=42)

    return train_x, test_x, train_y, test_y
