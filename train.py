import os
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train():
    # Set MLflow tracking
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(os.environ.get('MLFLOW_EXPERIMENT_NAME'))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args = parser.parse_args()
    
    # Load data
    train_df = pd.read_csv(f"{args.train}/train.csv")
    val_df = pd.read_csv(f"{args.validation}/validation.csv")
    
    X_train = train_df.drop('Churn', axis=1)
    y_train = train_df['Churn']
    X_val = val_df.drop('Churn', axis=1)
    y_val = val_df['Churn']
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "seed": 42
    }
    
    # MLflow tracking
    with mlflow.start_run():
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dval, "validation")],
            early_stopping_rounds=10
        )
        
        # Evaluate
        preds = model.predict(dval)
        preds_binary = [1 if p > 0.5 else 0 for p in preds]
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_val, preds_binary),
            "precision": precision_score(y_val, preds_binary),
            "recall": recall_score(y_val, preds_binary),
            "f1": f1_score(y_val, preds_binary)
        })
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        # Save model for SageMaker
        model.save_model(os.path.join(args.model_dir, "xgboost-model"))

if __name__ == "__main__":
    train()