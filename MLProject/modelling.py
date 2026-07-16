import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import mlflow
import mlflow.sklearn
import os
import json
import logging
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("modelling_tuning.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ['Age', 'Gender', 'Hydration_Level', 'Oil_Level', 'Sensitivity', 'Humidity', 'Temperature']

def load_preprocessed_data(data_path):
    logger.info(f"Loading preprocessed data from: {data_path}")
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS]
    y = df['Skin_Type_Encoded']
    logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

def perform_hyperparameter_tuning(X_train, y_train):
    logger.info("Starting hyperparameter tuning with GridSearchCV...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    y_pred_proba = model.predict_proba(X_test)
    fpr, tpr, roc_auc_dict = {}, {}, {}
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc_dict[i] = auc(fpr[i], tpr[i])
    mean_roc_auc = np.mean(list(roc_auc_dict.values()))
    logger.info(f"Mean ROC AUC: {mean_roc_auc:.4f}")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1,
            "mean_roc_auc": mean_roc_auc, "roc_auc_per_class": roc_auc_dict,
            "classification_report": report, "confusion_matrix": cm.tolist()}

def manual_logging(run_name, model, metrics, best_params, best_cv_score, X_train, y_train, X_test, y_test):
    logger.info(f"Starting MLflow manual logging: {run_name}")
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1_score"])
        mlflow.log_metric("mean_roc_auc", metrics["mean_roc_auc"])
        mlflow.log_metric("best_cv_score", best_cv_score)
        for class_idx, auc_val in metrics["roc_auc_per_class"].items():
            mlflow.log_metric(f"roc_auc_class_{class_idx}", auc_val)
        mlflow.sklearn.log_model(model, "model")
        logger.info("Model artifact logged.")
        report_path = "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(metrics["classification_report"], f, indent=2)
        mlflow.log_artifact(report_path)
        cm_path = "confusion_matrix.json"
        with open(cm_path, "w") as f:
            json.dump(metrics["confusion_matrix"], f, indent=2)
        mlflow.log_artifact(cm_path)
        model_path = "model_tuning.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        params_path = "best_params.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        mlflow.log_artifact(params_path)
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        for p in [report_path, cm_path, model_path, params_path]:
            if os.path.exists(p):
                os.remove(p)
    return run_id

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "skin_type_preprocessing.csv")
    X, y = load_preprocessed_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    mlflow.set_experiment("SkinType_Classification_Tuning")
    best_model, best_params, best_cv_score = perform_hyperparameter_tuning(X_train, y_train)
    metrics = evaluate_model(best_model, X_test, y_test)
    run_id = manual_logging(run_name="tuned_manual_logging", model=best_model, metrics=metrics,
        best_params=best_params, best_cv_score=best_cv_score,
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    logger.info("Modelling with hyperparameter tuning completed successfully!")
    logger.info(f"Final Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Final F1 Score: {metrics['f1_score']:.4f}")
