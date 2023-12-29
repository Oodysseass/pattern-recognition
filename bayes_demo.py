import numpy as np

from bayes_functions import estimate_means, test_bayes, plot_bayes
from data import load_data


train_x, test_x, train_y, test_y = load_data("dataset.csv", 0.5)

train_data = np.concatenate([train_x, train_y.reshape(-1, 1)], axis=1)
test_data = np.concatenate([test_x, test_y.reshape(-1, 1)], axis=1)
num_features = train_x.shape[1]
num_classes = max(train_y)


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
misses = len(pred[pred != test_y])

print("Misclassified:", misses, "samples")
print(f"Accuracy: {(len(test_y) - misses) / len(test_y): .2f}%")


## different covariance matrix for each class
cov_mats = {}
for i in range(num_classes):
    cov_mats[i] = np.cov(train_x[train_y == i + 1].T)

# test model
pred = test_bayes(p, means, cov_mats, test_x, num_classes)
plot_bayes(test_x, test_y, pred, p, means, cov_mats)
misses = len(pred[pred != test_y])

print("Misclassified:", len(pred[pred != test_y]), "samples")
print(f"Accuracy: {(len(test_y) - misses) / len(test_y): .2f}%")
