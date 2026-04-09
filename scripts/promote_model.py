from mlflow.tracking import MlflowClient

def promote_champion(model_name: str):
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