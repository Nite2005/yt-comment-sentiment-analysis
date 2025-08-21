import mlflow
import pytest
import pandas as pd
import pickle
from mlflow.tracking import MlflowClient

# Set your remote tracking URI
mlflow.set_tracking_uri("http://ec2-3-110-216-184.ap-south-1.compute.amazonaws.com:5000/")

@pytest.mark.parametrize("model_name, stage, vectorizer_path", [
    ("yt_chrome_plugin_model", "Staging", "tfidf_vectorizer.pkl"),  # Adjust paths & stage as needed
])
def test_model_with_vectorizer(model_name, stage, vectorizer_path):
    client = MlflowClient()

    # Get the latest version in the specified stage
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    assert latest_version_info, f"No model found in the '{stage}' stage for '{model_name}'"

    latest_version = latest_version_info[0].version

    try:
        # Load the latest version of the model
        model_uri = f"models:/{model_name}/{latest_version}"
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

        print(f"✅ Model '{model_name}' version {latest_version} successfully processed the dummy input from '{stage}' stage.")

    except Exception as e:
        pytest.fail(f"❌ Model test failed with error: {e}")
