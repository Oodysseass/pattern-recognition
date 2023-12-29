# -*- coding: utf-8 -*-
"""K-NN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Enu1Y_3dP3mNXdYA26T5gjwqwLSxaZ2J
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import mahalanobis
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools
from math import pi
import seaborn as sns

#Loading data
custom_columns = ['feature 1', 'feature 2', 'label']
data_file = pd.read_csv("./dataset.csv", header=None, names=custom_columns)
data_file = data_file.dropna()

data_file

#K-Nearest Neighboor
data_x = data_file.iloc[:, [0, 1]].values
data_y = data_file.iloc[:, 2].values
num_features = data_x.shape[1]
num_classes = max(data_y)
print (data_x)
print (data_y)
print (num_features)
print (num_classes)

#Spliting the database 50/50
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.5, stratify= data_y, random_state=42)

for k in range (10):

  #K-NN to the training set
  classifier = KNeighborsClassifier(n_neighbors=k+1, weights= 'uniform')
  classifier.fit(train_x, train_y)

  #predict test results
  pred_y = classifier.predict(test_x)
  print ('pred_y for k={}'.format(k+1), pred_y)
  print ('test_y', test_y)

##Making Confusion Matrix
  cm = confusion_matrix( test_y, pred_y)
  print (cm)

  cm_df = pd.DataFrame(cm,
                     index = ['1','2','3'],
                     columns = ['1','2','3'])
  #Plots
  #Plotting the confusion matrix
  plt.figure(figsize=(5,4))
  sns.heatmap(cm_df, annot=True)
  plt.title('Confusion Matrix')
  plt.ylabel('Actal Values')
  plt.xlabel('Predicted Values')
  plt.show()

  #Plotting boundary regions
  fig = plot_decision_regions(test_x, test_y, clf=classifier, legend=2)
  plt.title('Decision Regions for k={} on Training Data'.format(k+1))
  plt.show()
