from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from preprocessing import build_preprocessor

NUMERICAL_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_COLS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
EXPERIMENT_NAME = 'Heart disease pipeline experiment'
CHAMPION_MODEL = 'DecisionTreeClassifier'

PIPELINES = {
        'DecisionTreeClassifier': (
            Pipeline([
                ('prep', build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS, strategy='tree')), 
                ('classifier', DecisionTreeClassifier())
        ]),{
            'classifier__criterion': 'gini',
            'classifier__min_samples_leaf': 1,
            'classifier__min_samples_split': 2,
            'classifier__splitter': 'random'
        }
    ),
        'RandomForestClassifier': (
            Pipeline([
                ('prep', build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS, strategy='tree')), 
                ('classifier', RandomForestClassifier())
        ]),{
            'classifier__criterion': 'entropy',
            'classifier__min_samples_leaf': 1,
            'classifier__min_samples_split': 5,
            'classifier__n_estimators': 40
        }
    ),
        'GradientBoostingClassifier': (
            Pipeline([
                ('prep', build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS, strategy='tree')), 
                ('classifier', GradientBoostingClassifier())
        ]), {
            'classifier__criterion': 'friedman_mse',
            'classifier__learning_rate': 0.1,
            'classifier__loss': 'log_loss',
            'classifier__n_estimators': 500
        }
    ),
        'KNN': (
            Pipeline([
                ('prep', build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS, strategy='knn')), 
                ('classifier', KNeighborsClassifier())
        ]),{
            'classifier__n_neighbors': 3,
            'classifier__p': 1
        }
    ),
        'LogisticRegression': (
            Pipeline([
                ('prep', build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS, strategy='linear')), 
                ('classifier', LogisticRegression(max_iter=2000))
        ]),{
            'classifier__C': 2.0,
            'classifier__solver': 'lbfgs',
            'classifier__tol': 0.0001
        }
    ),
        'SVC': (
            Pipeline([
                ('prep', build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS, strategy='linear')), 
                ('classifier', SVC())
        ]),{
            'classifier__C': 2.0,
            'classifier__kernel': 'poly',
            'classifier__tol': 0.0001,
            'classifier__probability': True
        }
    ),
        'MLPC': (
        Pipeline([
            ('prep', build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS, strategy='linear')), 
            ('classifier', MLPClassifier(max_iter=3000))
        ]),{
            'classifier__activation': 'tanh',
            'classifier__batch_size': 10,
            'classifier__solver': 'adam'
        }
    ) 
}

PIPELINES_FOR_TUNNING = {
        'DecisionTreeClassifier': (
            Pipeline([
                ('prep', build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS, strategy='tree')), 
                ('classifier', DecisionTreeClassifier())
        ]),{
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__splitter': ['best', 'random'],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 5, 10]
        }
    ),
        'RandomForestClassifier': (
            Pipeline([
                ('prep', build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS, strategy='tree')), 
                ('classifier', RandomForestClassifier())
        ]),{
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__n_estimators': [10, 40, 100, 150],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 5, 10]
        }
    ),
        'GradientBoostingClassifier': (
            Pipeline([
                ('prep', build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS, strategy='tree')), 
                ('classifier', GradientBoostingClassifier())
        ]), {
            'classifier__loss': ['log_loss', 'exponential'],
            'classifier__learning_rate': [0.1, 0.01, 0.001, 0.0001],
            'classifier__n_estimators': [25, 50, 100, 200, 300, 500],
            'classifier__criterion': ['friedman_mse', 'squared_error']
        }
    ),
        'KNN': (
            Pipeline([
                ('prep', build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS, strategy='knn')), 
                ('classifier', KNeighborsClassifier())
        ]),{
            'classifier__n_neighbors': [3, 5, 10, 20],
            'classifier__p': [1, 2]
        }
    ),
        'LogisticRegression': (
            Pipeline([
                ('prep', build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS, strategy='linear')), 
                ('classifier', LogisticRegression(max_iter=2000))
        ]),{
            'classifier__tol': [0.0001, 0.00001, 0.000001],
            'classifier__C': [1.0, 1.5, 2.0],
            'classifier__solver': ['lbfgs', 'sag', 'saga']
        }
    ),
        'SVC': (
            Pipeline([
                ('prep', build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS, strategy='linear')), 
                ('classifier', SVC(probability=True))
        ]),{
            'classifier__tol': [0.001, 0.0001, 0.00001],
            'classifier__C': [1.0, 1.5, 2.0],
            'classifier__kernel': ['rbf', 'linear', 'poly', 'sigmoid']
        }
    ),
        'MLPC': (
        Pipeline([
            ('prep', build_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS, strategy='linear')), 
            ('classifier', MLPClassifier(max_iter=3500))
        ]),{
            'classifier__activation': ['relu', 'logistic', 'tanh'],
            'classifier__solver': ['adam', 'sgd'],
            'classifier__batch_size': [10, 56]
        }
    ) 
}