from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

def build_preprocessor(numerical_cols: list, categorical_cols: list, strategy: str = 'tree') -> ColumnTransformer:

    if strategy == 'tree':
        return ColumnTransformer(
            transformers=[
                ('categorical', OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1  # needed when handle_unknown='use_encoded_value'
                ), categorical_cols),
                ('numerical', 'passthrough', numerical_cols)
            ])

    elif strategy == 'knn':
        return ColumnTransformer(
            transformers=[
                ('categorical', OneHotEncoder(
                    handle_unknown='ignore'), categorical_cols),
                ('numerical', StandardScaler(), numerical_cols)
                ])

    elif strategy == 'linear':
        return ColumnTransformer(
            transformers=[
                ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                ('numerical', StandardScaler(), numerical_cols)
            ])
