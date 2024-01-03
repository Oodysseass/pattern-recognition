import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filename, size):
    ## load data
    data_file = pd.read_csv(filename, header=None)
    data_file = data_file.dropna()

    # make train and test sets
    data_x = np.array(data_file.drop(data_file.columns[-1], axis=1))
    data_y = np.array(data_file[data_file.columns[-1]])

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,
                                                        test_size=size,
                                                        stratify=data_y,
                                                        random_state=42)

    return train_x, test_x, train_y, test_y
