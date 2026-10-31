
# ==========================================
# 1. Import required libraries
# ==========================================

import os
import pandas as pd
import xgboost as xgb
import mlflow
import joblib


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path


# ==========================================
# 2. Load training and testing data
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

Xtrain = pd.read_csv(PROJECT_DIR / "Xtrain.csv")
Xtest = pd.read_csv(PROJECT_DIR / "Xtest.csv")
ytrain = pd.read_csv(PROJECT_DIR / "ytrain.csv").squeeze()
ytest = pd.read_csv(PROJECT_DIR / "ytest.csv").squeeze()

print("Training data:", Xtrain.shape)
print("Testing data:", Xtest.shape)


# ==========================================
# 3. Identify numeric and categorical features
# ==========================================

numeric_features = Xtrain.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = Xtrain.select_dtypes(
    exclude=["number"]
).columns.tolist()


# ==========================================
# 4. Calculate class imbalance ratio
# ==========================================

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

print("Class weight:", class_weight)


# ==========================================
# 5. Define preprocessing
# ==========================================

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)


# ==========================================
# 6. Define XGBoost model
# ==========================================

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    n_jobs=1,
    eval_metric="logloss"
)


# ==========================================
# 7. Define hyperparameter grid
# ==========================================

param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__colsample_bytree": [0.4, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.6]
}


# ==========================================
# 8. Create model pipeline
# ==========================================

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)


# ==========================================
# 9. Configure MLflow experiment
# ==========================================

mlflow.set_experiment("Wellness_Tourism_Prediction")


# ==========================================
# 10. Hyperparameter tuning and experiment tracking
# ==========================================

with mlflow.start_run(run_name="XGBoost_GridSearch"):

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    print("\nStarting hyperparameter tuning...")

    grid_search.fit(Xtrain, ytrain)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print("\nBest cross-validation F1-score:")
    print(grid_search.best_score_)

    # Log all parameter combinations
    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        with mlflow.start_run(nested=True):

            mlflow.log_params(param_set)

            mlflow.log_metric("mean_test_f1", mean_score)
            mlflow.log_metric("std_test_f1", std_score)

    # Log best parameters in parent run
    mlflow.log_params(grid_search.best_params_)

    # Retrieve the best model
    best_model = grid_search.best_estimator_


    # ======================================
    # 11. Evaluate the best model
    # ======================================

    classification_threshold = 0.45

    # Training predictions
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]

    y_pred_train = (
        y_pred_train_proba >= classification_threshold
    ).astype(int)

    # Testing predictions
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

    y_pred_test = (
        y_pred_test_proba >= classification_threshold
    ).astype(int)

    # Generate classification reports
    train_report = classification_report(
        ytrain,
        y_pred_train,
        output_dict=True,
        zero_division=0
    )

    test_report = classification_report(
        ytest,
        y_pred_test,
        output_dict=True,
        zero_division=0
    )

    # Print evaluation results
    print("\nTRAINING PERFORMANCE")
    print(classification_report(ytrain, y_pred_train, zero_division=0))

    print("\nTESTING PERFORMANCE")
    print(classification_report(ytest, y_pred_test, zero_division=0))

    # Calculate ROC-AUC
    train_roc_auc = roc_auc_score(ytrain, y_pred_train_proba)
    test_roc_auc = roc_auc_score(ytest, y_pred_test_proba)

    # Log evaluation metrics to MLflow
    mlflow.log_metrics({

        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1_score": train_report["1"]["f1-score"],
        "train_roc_auc": train_roc_auc,

        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1_score": test_report["1"]["f1-score"],
        "test_roc_auc": test_roc_auc

    })


# ==========================================
# 12. Save the best trained model
# ==========================================

MODEL_DIR = PROJECT_DIR / "deployment"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model.joblib"
joblib.dump(best_model, MODEL_PATH)

print("\nMODEL SAVED SUCCESSFULLY")
print(f"Location: {MODEL_PATH}")
