import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from bayes_functions import estimate_means, test_bayes, plot_bayes


## load data
custom_columns = ['feature 1', 'feature 2', 'label']
data_file = pd.read_csv("./dataset.csv", header=None, names=custom_columns)
data_file = data_file.dropna()

#data_file['label'].plot(kind='hist', bins=[0.75, 1.25, 1.75, 2.25, 2.75, 3.25])
#plt.show()

# make train and test sets
data_x = np.array(data_file.drop('label', axis=1))
data_y = np.array(data_file['label'])
num_features = data_x.shape[1]
num_classes = max(data_y)

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,
                                                    test_size=0.5,
                                                    stratify=data_y,
                                                    random_state=42)

train_data = np.concatenate([train_x, train_y.reshape(-1, 1)], axis=1)
test_data = np.concatenate([test_x, test_y.reshape(-1, 1)], axis=1)


## maximum likelihood
# theta_1 = mean 
means = estimate_means(num_classes, num_features, train_data)

# theta_2 = covariance matrix, same covariance for all classes
cov_mats = {}
for i in range(num_classes):
    cov_mats[i] = np.cov(train_x.T)


## bayes model
# a priori
p = np.ones(num_classes)
for i in range(num_classes): 
    p[i] = len(train_data[train_y == i + 1]) / len(train_data)

# test model
pred = test_bayes(p, means, cov_mats, test_x, num_classes)
plot_bayes(test_x, test_y, pred, p, means, cov_mats)


## different covariance matrix for each class
cov_mats = {}
for i in range(num_classes):
    cov_mats[i] = np.cov(train_x[train_y == i + 1].T)

# test model
pred = test_bayes(p, means, cov_mats, test_x, num_classes)
plot_bayes(test_x, test_y, pred, p, means, cov_mats)

