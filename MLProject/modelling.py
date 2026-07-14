import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import mlflow
import mlflow.sklearn
import os
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("modelling_ci.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_preprocessed_data(data_path):
    logger.info(f"Loading preprocessed data from: {data_path}")
    df = pd.read_csv(data_path)
    feature_columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    X = df[feature_columns]
    y = df["target"]
    logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def perform_hyperparameter_tuning(X_train, y_train):
    logger.info("Starting hyperparameter tuning...")
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
    return metrics, classification_report(y_test, y_pred, output_dict=True), confusion_matrix(y_test, y_pred).tolist()


if __name__ == "__main__":
    data_path = os.environ.get("DATA_PATH", "iris_preprocessing.csv")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    X, y = load_preprocessed_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Iris_Classification_CI")

    best_model, best_params, best_cv_score = perform_hyperparameter_tuning(X_train, y_train)
    metrics, report, cm = evaluate_model(best_model, X_test, y_test)

    with mlflow.start_run(run_name="ci_retraining"):
        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("model_type", "RandomForestClassifier")

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.log_metric("best_cv_score", best_cv_score)

        mlflow.sklearn.log_model(best_model, "model")

        report_path = "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)

        cm_path = "confusion_matrix.json"
        with open(cm_path, "w") as f:
            json.dump(cm, f, indent=2)
        mlflow.log_artifact(cm_path)

        run_id = mlflow.active_run().info.run_id
        logger.info(f"CI Training completed! Run ID: {run_id}")
        logger.info(f"Final Accuracy: {metrics['accuracy']:.4f}")

        if os.path.exists(report_path):
            os.remove(report_path)
        if os.path.exists(cm_path):
            os.remove(cm_path)
