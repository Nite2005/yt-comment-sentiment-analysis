import mlflow
from mlflow.tracking import MlflowClient

def promote_model():
    # Set up AWS MLflow tracking URI
    mlflow.set_tracking_uri("http://ec2-3-110-216-184.ap-south-1.compute.amazonaws.com:5000/")

    client = MlflowClient()
    model_name = "yt_chrome_plugin_model"

    # Get the version by alias "staging"
    staging_version = client.get_model_version_by_alias(model_name, "Staging")
    if not staging_version:
        raise ValueError(f"No model found with alias 'staging' for {model_name}")
    staging_version_number = staging_version.version

    # Archive current production version (if alias exists)
    try:
        prod_version = client.get_model_version_by_alias(model_name, "Production")
        if prod_version:
            client.transition_model_version_stage(
                name=model_name,
                version=prod_version.version,
                stage="Archived"
            )
            print(f"Archived Production model version {prod_version.version}")
    except Exception:
        print("⚠️ No existing Production alias found, skipping archive.")

    # Promote staging → production
    client.transition_model_version_stage(
        name=model_name,
        version=staging_version_number,
        alias="Production"
    )

    # Update alias for production
    client.set_registered_model_alias(model_name, "Production", staging_version_number)

    print(f"✅ Model version {staging_version_number} promoted from @staging → @production")

if __name__ == "__main__":
    promote_model()
