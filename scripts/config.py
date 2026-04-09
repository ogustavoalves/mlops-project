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

