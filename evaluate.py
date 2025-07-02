import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate(test_path, model_path):
    df = pd.read_csv(test_path)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    booster = xgb.Booster()
    booster.load_model(model_path)
    
    dtest = xgb.DMatrix(X)
    preds = booster.predict(dtest)
    preds_binary = [1 if p > 0.5 else 0 for p in preds]
    
    acc = accuracy_score(y, preds_binary)
    prec = precision_score(y, preds_binary)
    rec = recall_score(y, preds_binary)
    
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")

if __name__ == "__main__":
    import sys
    test_path = sys.argv[1]
    model_path = sys.argv[2]
    evaluate(test_path, model_path)
