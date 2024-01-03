# -*- coding: utf-8 -*-
"""K-NN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Enu1Y_3dP3mNXdYA26T5gjwqwLSxaZ2J
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay

from data import load_data


# load data
train_x, test_x, train_y, test_y = load_data("dataset.csv", 0.5)

train_data = np.concatenate([train_x, train_y.reshape(-1, 1)], axis=1)
test_data = np.concatenate([test_x, test_y.reshape(-1, 1)], axis=1)

fig, axs = plt.subplots(5, 4, figsize=(10, 10), layout='tight')
row, col = [0, 0]
for k in range(10):
  ## K-NN to the training set
  classifier = KNeighborsClassifier(n_neighbors=k+1, weights='uniform')
  classifier.fit(train_x, train_y)

  # predict test results
  pred_y = classifier.predict(test_x)
  acc = accuracy_score(test_y, pred_y)

  ## Plots
  # Plotting the confusion matrix
  ConfusionMatrixDisplay.from_predictions(test_y, pred_y, ax=axs[row, col])
  axs[row, col].set_title('Confusion Matrix')
  col += 1

  # Plotting boundary regions
  DecisionBoundaryDisplay.from_estimator(classifier, train_x, ax=axs[row, col])
  axs[row, col].scatter(test_x[:, 0], test_x[:, 1], s=25, c=test_y, edgecolor='k')
  axs[row, col].set_title(f'k={k + 1}, error={100 * (1 - acc):.2f}%')

  col = (col + 1) % 4
  row = row + 1 if col == 0 else row

plt.show()
