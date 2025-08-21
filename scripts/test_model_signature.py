import mlflow
import pytest
import pandas as pd
import pickle
from mlflow.tracking import MlflowClient

# Set your remote tracking URI
mlflow.set_tracking_uri("http://ec2-3-110-216-184.ap-south-1.compute.amazonaws.com:5000/")

@pytest.mark.parametrize("model_name, alias, vectorizer_path", [
    ("yt_chrome_plugin_model", "staging", "tfidf_vectorizer.pkl"),  # Adjust paths & alias as needed
])
def test_model_with_vectorizer(model_name, alias, vectorizer_path):
    client = MlflowClient()

    # Get model version by alias
    version_info = client.get_model_version_by_alias(model_name, alias)
    assert version_info is not None, f"No model found with alias '{alias}' for '{model_name}'"

    try:
        # Load the model using alias
        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Load the vectorizer
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        # Create a dummy input for the model
        input_text = "hi how are you"
        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # Predict using the model
        prediction = model.predict(input_df)

        # Verify the input shape matches the vectorizer's feature output
        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), "Input feature count mismatch"

        # Verify output shape (predictions should match input rows)
        assert len(prediction) == input_df.shape[0], "Output row count mismatch"

        print(f"✅ Model '{model_name}' (alias: {alias}, version: {version_info.version}) successfully processed the dummy input.")

    except Exception as e:
        pytest.fail(f"❌ Model test failed with error: {e}")
