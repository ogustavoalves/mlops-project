import pandas as pd
from sklearn.model_selection import train_test_split
from model_tuner import generic_grid_search_cv
from mlflow_manager import train_and_log_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import mlflow
# Leitura da base
dataset_path = '../data/raw/heart.csv'
dataset = pd.read_csv(dataset_path)

# Definir o tracking ui
mlflow.set_tracking_uri(uri='http://localhost:5000')
# Definindo o experimento do registro
response = mlflow.set_experiment('Heart disease experiment')

X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, random_state=0) 



# Tunning de Hiperparâmetros
# Dicionários de parâmetros e valores para efetuar Cross validation
# Decision tree
tree_params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

# Random florest
rf_params = {
    'criterion': ['gini', 'entropy'],
    'n_estimators': [10, 40, 100, 150],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

# Gradient boosting classifier
gbc_params = {
    'loss': ['log_loss', 'exponential'],
    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
    'n_estimators': [25, 50, 100, 200, 300, 500],
    'criterion': ['friedman_mse', 'squared_error']
}

# K-nearest neighbors classifier
knn_params = {
    'n_neighbors': [3, 5, 10, 20],
    'p': [1, 2]
}

# Logistic regression
lr_params = {
    'tol': [0.0001, 0.00001, 0.000001],
    'C': [1.0, 1.5, 2.0],
    'solver': ['lbfgs', 'sag', 'saga']
}

# Support vector machine
svm_params = {
    'tol': [0.001, 0.0001, 0.00001],
    'C': [1.0, 1.5, 2.0],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

# Neural network classifier
neural_net_params = {
    'activation': ['relu', 'logistic', 'tahn'],
    'solver': ['adam', 'sgd'],
    'batch_size': [10, 56]
}

all_params = {
    'tree_params': [tree_params, DecisionTreeClassifier()],
    'rf_params': [rf_params, RandomForestClassifier()],
    'gbc_params': [gbc_params, GradientBoostingClassifier()],
    'knn_params': [knn_params, KNeighborsClassifier()],
    'lr_params': [lr_params, LogisticRegression()],
    'svm_params': [svm_params, SVC()],
    'neural_net_params': [neural_net_params, MLPClassifier()],
}


best_params_dict = {}
for name, param in all_params.items():
    name, best_params = generic_grid_search_cv(estimator_classifier=param[1], param_dict=param[0], X_matrix=X, y_matrix=y)
    # os values provavelmente tem de ser salvos numa lista onde 0 = best_params e 1 = objeto
    best_params_dict[f'{name}_best_params'] = best_params, param[1]

# Treinamento 
# Registro MLflow


for name, param in best_params_dict.items():
    print(f'model: {name}')
    model_tacking = train_and_log_model(param[1], X_train, X_test, y_train, y_test, param[0], dataset, dataset_path)

    print(f'artifact path: {model_tacking.artifact_path}')