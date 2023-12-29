import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from data import load_data


train_x, test_x, train_y, test_y = load_data("dataset.csv", 0.5)

train_data = np.concatenate([train_x, train_y.reshape(-1, 1)], axis=1)
test_data = np.concatenate([test_x, test_y.reshape(-1, 1)], axis=1)


# linear svm classifier
svm = SVC(kernel='linear')
svm.fit(train_x, train_y)

pred = svm.predict(test_x)

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
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
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
print(f"Accuracy: {accuracy * 100:.2f}%")
