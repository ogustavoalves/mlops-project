from mlflow.tracking import MlflowClient

def promote_champion(model_name: str):
    """
    Promote the latest version of a registered model to the Production stage.
    This function retrieves the most recent version of a specified MLflow model
    and transitions it to the Production stage, automatically archiving any
    existing versions already in Production.
    Args:
        model_name (str): The name of the registered model to promote.
    Returns:
        None
    Raises:
        Prints error message if model promotion fails, including:
        - Model not found in registry
        - Invalid model name
        - MLflow client connection issues
    Example:
        >>> promote_champion('my_ml_model')
        Model my_ml_model v3 moved to Production
    """
    try:
        client = MlflowClient()
        latest_versions = client.search_model_versions(f"name='{model_name}'")
        model_version = latest_versions[0].version
        
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage='Production',
            archive_existing_versions=True
        )
        
        print(f'Model {model_name} v{model_version} moved to Production')
        
    except Exception as e:
        print(f'Error promoting model: {e}')