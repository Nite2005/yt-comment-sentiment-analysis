import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient

# Set your remote tracking URI
mlflow.set_tracking_uri("http://ec2-3-110-216-184.ap-south-1.compute.amazonaws.com:5000/")

@pytest.mark.parametrize("model_name, alias", [
    ("yt_chrome_plugin_model", "Staging"),  # use lowercase aliases by convention
])
def test_load_model_by_alias(model_name, alias):
    client = MlflowClient()

    # Get model version by alias
    version_info = client.get_model_version_by_alias(model_name, alias)
    assert version_info is not None, f"No model found with alias '{alias}' for '{model_name}'"

    try:
        # Load the model by alias
        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Ensure the model loads successfully
        assert model is not None, "Model failed to load"
        print(f"✅ Model '{model_name}' version {version_info.version} loaded successfully using alias '{alias}'.")

    except Exception as e:
        pytest.fail(f"❌ Model loading failed with error: {e}")
