import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from data import load_data


def plot_svm(model, train_x, test_x, test_y, pred):
    support = model.support_vectors_
    matches = test_y == pred
    mismatches = test_y != pred
    colors = {
        'train': 'royalblue',
        'test-success': 'purple',
        'support': 'green',
        'test-fail': 'red'
    }

    DecisionBoundaryDisplay.from_estimator(model, train_x)
    plt.scatter(train_x[:, 0], train_x[:, 1], color='royalblue', edgecolors='k')
    plt.scatter(test_x[matches, 0], test_x[matches, 1], color='purple', edgecolors='k')
    plt.scatter(support[:, 0], support[:, 1], color='green', edgecolors='k')
    plt.scatter(test_x[mismatches, 0], test_x[mismatches, 1], color='red', edgecolors='k')

    legend_elements = [Line2D([0], [0], marker='o', color='w', \
                       markerfacecolor=colors[label], markersize=10, \
                       label=f'{label}') for label in colors]

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(handles=legend_elements)

    plt.show()
    return


train_x, test_x, train_y, test_y = load_data("dataset.csv", 0.5)

train_data = np.concatenate([train_x, train_y.reshape(-1, 1)], axis=1)
test_data = np.concatenate([test_x, test_y.reshape(-1, 1)], axis=1)


# linear svm classifier
svm = SVC(kernel='linear')
svm.fit(train_x, train_y)

pred = svm.predict(test_x)
plot_svm(svm, train_x, test_x, test_y, pred)

accuracy = accuracy_score(test_y, pred)
misses = len(pred[pred != test_y])
print("|-------------Linear SVM-------------|")
print("Misclassified:", len(pred[pred != test_y]), "samples")
print(f"Accuracy: {accuracy * 100:.2f}%\n")


## RBF svm
print("|---------------SVM RBF--------------|")
temp_svm = SVC(kernel='rbf')

# hyper-parameters
parameters = {
    'C': [0.1, 1.0, 10.0, 100.0],
    'gamma': ['auto', 0.01, 0.1, 1, 10],
}

# cv
grid = GridSearchCV(temp_svm, parameters, cv=10)
grid.fit(train_x, train_y)
print("Best parameters:", grid.best_params_)

# predict
svm_rbf = grid.best_estimator_

pred = svm_rbf.predict(test_x)

accuracy = accuracy_score(test_y, pred)
misses = len(pred[pred != test_y])
print("Misclassified:", len(pred[pred != test_y]), "samples")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
plot_svm(svm_rbf, train_x, test_x, test_y, pred)


print("|---------------DEFAULT SVM--------------|")
default_svm = SVC(kernel='rbf', gamma='auto')

default_svm.fit(train_x, train_y)
pred = default_svm.predict(test_x)

accuracy = accuracy_score(test_y, pred)
misses = len(pred[pred != test_y])
print("Misclassified:", len(pred[pred != test_y]), "samples")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
plot_svm(default_svm, train_x, test_x, test_y, pred)
