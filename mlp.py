from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from data import load_data


# data loading and preprocessing
train_x, test_x, train_y, test_y = load_data("datasetC.csv", 0.2)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# cv
#param_grid = {
#    'hidden_layer_sizes': [(64, 32), (128, 64), (32, 32, 32)],
#    'activation': ['relu', 'tanh', 'logistic'],
#    'solver': ['adam', 'lbfgs', 'sgd'],
#    'batch_size': [32, 64, 128]
#}
#
#model = MLPClassifier(random_state=42)
#grid_search = GridSearchCV(model, param_grid, cv=10, verbose=10)
#grid_search.fit(train_x, train_y)
#
#print(f'Best params: {grid_search.best_params_}')

# best classifier
#mlp = grid_search.best_estimator_
mlp = MLPClassifier(hidden_layer_sizes=(256, 128),
                    activation='relu',
                    batch_size=32,
                    solver='adam',
                    random_state=42,
                    verbose=10,
                    n_iter_no_change=300,
                    max_iter=300)

# train
mlp.fit(train_x, train_y)

# test
pred_y = mlp.predict(test_x)
accuracy = accuracy_score(test_y, pred_y)
print(f"Accuracy: {accuracy * 100:.2f}%")
