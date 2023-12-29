import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis
from math import pi


# calculate p(x|w_i) * p(w_i) with assumed normal of p(x|w_i)
def bayes(sample, priori, mean, cov):
    distance = mahalanobis(sample, mean, np.linalg.inv(cov))
    p_x_i = 1 / (2 * pi * np.linalg.det(cov) ** (1 / 2)) * np.exp(- 1 / 2 * distance)

    return p_x_i * priori

# calculate mean of each class
def estimate_means(num_classes, num_features, train_data):
    train_x = train_data[:, 0:-1]
    train_y = train_data[:, -1]
    means = np.ones((num_classes, num_features))

    for i in range(num_classes): 
        means[i, :] = np.mean(train_x[train_y == i + 1], axis=0)

    return means

# tests bayes model
def test_bayes(p, means, cov_mats, test_x, num_classes):
    pred_y = np.zeros(test_x.shape[0])

    for i in range(len(pred_y)):
        max_p = 0
        for j in range(num_classes):
            p_j = bayes(test_x[i, :], p[j], means[j, :], cov_mats[j])
            if max_p < p_j:
                max_p = p_j
                pred_y[i] = j + 1

    return pred_y

# plot decision regions and wrongs
def plot_bayes(test_x, test_y, pred, p, means, cov_mats):
    ## get mismatches
    mismatches = test_y != pred
    pred[mismatches] = 0
    colors = {1: 'green', 2: 'purple', 3: 'yellow', 0: 'red'}
    
    ## plot decision regions
    h = .05
    x_min, x_max = test_x[:, 0].min() - 1, test_x[:, 0].max() + 1
    y_min, y_max = test_x[:, 1].min() - 1, test_x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    temp_x = np.stack((xx.flatten(), yy.flatten()), axis=0).T

    zz = test_bayes(p, means, cov_mats, temp_x, 3)
    plt.scatter(temp_x[:, 0], temp_x[:, 1], c=[colors[label] for label in zz])

    ## plot results
    plt.scatter(test_x[:, 0], test_x[:, 1], c=[colors[label] for label in pred], marker='o', edgecolors='k')

    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=10, label=f'Class {label}') for label in [1, 2, 3]]
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Missclassified'))

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(handles=legend_elements)

    plt.show()


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

