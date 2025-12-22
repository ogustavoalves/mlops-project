from sklearn.model_selection import GridSearchCV

def generic_grid_search_cv(estimator_classifier, param_dict, X_matrix, y_matrix):
    """
    Executa um GridSearchCV para um classificador do Scikit-Learn.

    Esta função recebe um estimador do Scikit-Learn (classificador),
    realiza uma busca em grade com validação cruzada (GridSearchCV) 
    utilizando os hiperparâmetros fornecidos e retorna o nome do 
    estimador e o melhor conjunto de hiperparâmetros encontrado.

    Args:
        estimator_classifier: Estimador Scikit-Learn, como 
            `DecisionTreeClassifier()`, `RandomForestClassifier()`, etc.
        param_dict (dict): Dicionário contendo hiperparâmetros como chaves
            e listas de valores como opções a serem testadas.
        X_matrix (numpy.ndarray): Matriz de features usada para o treinamento.
        y_matrix (numpy.ndarray): Vetor ou matriz de rótulos alvo.

    Returns:
        estimator_name (str): Nome da classe do estimador recebido.
        best_params (dict): Melhor combinação de hiperparâmetros encontrada 
            pelo GridSearchCV.
    """
    
    grid_search = GridSearchCV(estimator=estimator_classifier, param_grid=param_dict)
    grid_search.fit(X = X_matrix, y = y_matrix)
    best_params = grid_search.best_params_
    estimator_name = estimator_classifier.__class__.__name__
    
    return estimator_name, best_params
