import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
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

