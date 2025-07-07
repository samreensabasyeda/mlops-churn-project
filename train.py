import os
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import argparse
import logging
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Configure MLflow tracking"""
    try:
        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
        mlflow.set_experiment(os.environ['MLFLOW_EXPERIMENT_NAME'])
        logger.info("MLflow tracking configured successfully")
    except KeyError as e:
        logger.error(f"Missing required environment variable: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error configuring MLflow: {str(e)}")
        raise

def load_data(train_path, val_path):
    """Load and validate training data"""
    try:
        logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path)
        
        logger.info(f"Loading validation data from {val_path}")
        val_df = pd.read_csv(val_path)
        
        # Validate data
        for df, name in [(train_df, "train"), (val_df, "validation")]:
            if 'Churn' not in df.columns:
                raise ValueError(f"Missing 'Churn' column in {name} data")
            if df.empty:
                raise ValueError(f"Empty {name} dataframe")
        
        return train_df, val_df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def train():
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
        parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
        parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
        args = parser.parse_args()
        
        # Paths
        train_path = os.path.join(args.train, "train.csv")
        val_path = os.path.join(args.validation, "validation.csv")
        
        # Load data
        train_df, val_df = load_data(train_path, val_path)
        
        # Prepare data
        X_train = train_df.drop('Churn', axis=1)
        y_train = train_df['Churn']
        X_val = val_df.drop('Churn', axis=1)
        y_val = val_df['Churn']
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "seed": 42
        }
        
        # Start MLflow run
        with mlflow.start_run() as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            
            # Train model
            logger.info("Starting model training...")
            start_time = datetime.now()
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, "train"), (dval, "validation")],
                early_stopping_rounds=10,
                verbose_eval=10
            )
            
            # Evaluate
            logger.info("Evaluating model...")
            preds = model.predict(dval)
            preds_binary = [1 if p > 0.5 else 0 for p in preds]
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_val, preds_binary),
                "precision": precision_score(y_val, preds_binary),
                "recall": recall_score(y_val, preds_binary),
                "f1": f1_score(y_val, preds_binary),
                "roc_auc": roc_auc_score(y_val, preds),
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            logger.info("Model metrics:")
            for k, v in metrics.items():
                logger.info(f"{k}: {v:.4f}")
            
            # Log confusion matrix
            cm = confusion_matrix(y_val, preds_binary)
            cm_dict = {
                "true_negative": int(cm[0][0]),
                "false_positive": int(cm[0][1]),
                "false_negative": int(cm[1][0]),
                "true_positive": int(cm[1][1])
            }
            mlflow.log_dict(cm_dict, "confusion_matrix.json")
            
            # Log model
            mlflow.xgboost.log_model(model, "model")
            logger.info("Model logged to MLflow")
            
            # Save model for SageMaker
            model_path = os.path.join(args.model_dir, "xgboost-model")
            model.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Training completed in {duration:.2f} seconds")
            
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    train()