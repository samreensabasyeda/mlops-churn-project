import os
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train(train_path, model_output_dir):
    # ✅ Set the MLflow Tracking URI
    mlflow.set_tracking_uri("http://13.203.193.28:30172/")

    # Create or set the experiment
    mlflow.set_experiment("ChurnPrediction")
    
    # Load and prepare data
    df = pd.read_csv(train_path)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "seed": 42
    }

    # 💡 Start MLflow run and log metrics
    with mlflow.start_run():
        booster = xgb.train(params, dtrain, num_boost_round=100)
        
        preds = booster.predict(dval)
        preds_binary = [1 if p > 0.5 else 0 for p in preds]
        acc = accuracy_score(y_val, preds_binary)
        
        mlflow.log_metric("val_accuracy", acc)
        mlflow.xgboost.log_model(booster, artifact_path="model")

        # Save model for SageMaker deployment
        model_path = os.path.join(model_output_dir, "xgboost-model")
        booster.save_model(model_path)

if __name__ == "__main__":
    import sys
    train_path = sys.argv[1]
    model_output_dir = sys.argv[2]
    train(train_path, model_output_dir)
