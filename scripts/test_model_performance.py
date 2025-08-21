import pytest
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.tracking import MlflowClient

# Set your remote tracking URI
mlflow.set_tracking_uri("http://ec2-3-110-216-184.ap-south-1.compute.amazonaws.com:5000/")

@pytest.mark.parametrize("model_name, alias, holdout_data_path, vectorizer_path", [
    ("yt_chrome_plugin_model", "staging", "data/interim/test_processed.csv", "tfidf_vectorizer.pkl"),  # Adjust paths
])
def test_model_performance(model_name, alias, holdout_data_path, vectorizer_path):
    try:
        client = MlflowClient()

        # Fetch model version using alias (new API)
        version_info = client.get_model_version_by_alias(model_name, alias)
        assert version_info is not None, f"No model found with alias '{alias}' for '{model_name}'"

        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Load the vectorizer
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        # Load holdout test data
        holdout_data = pd.read_csv(holdout_data_path)
        X_holdout_raw = holdout_data.iloc[:, 0].fillna("")  # First column = text
        y_holdout = holdout_data.iloc[:, -1]               # Last column = labels

        # Apply TF-IDF transformation
        X_holdout_tfidf = vectorizer.transform(X_holdout_raw)
        X_holdout_tfidf_df = pd.DataFrame(
            X_holdout_tfidf.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # Predict using the model
        y_pred_new = model.predict(X_holdout_tfidf_df)

        # Calculate performance metrics
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new, average='weighted', zero_division=1)
        recall_new = recall_score(y_holdout, y_pred_new, average='weighted', zero_division=1)
        f1_new = f1_score(y_holdout, y_pred_new, average='weighted', zero_division=1)

        # Define expected thresholds
        expected_threshold = 0.40
        assert accuracy_new >= expected_threshold, f'Accuracy {accuracy_new:.2f} < {expected_threshold}'
        assert precision_new >= expected_threshold, f'Precision {precision_new:.2f} < {expected_threshold}'
        assert recall_new >= expected_threshold, f'Recall {recall_new:.2f} < {expected_threshold}'
        assert f1_new >= expected_threshold, f'F1 {f1_new:.2f} < {expected_threshold}'

        print(f"✅ Performance test passed for model '{model_name}' (alias: {alias}, version: {version_info.version})")

    except Exception as e:
        pytest.fail(f"❌ Model performance test failed with error: {e}")
