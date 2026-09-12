import json, os, tarfile
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import average_precision_score

MODEL = "/opt/ml/processing/model"
TEST  = "/opt/ml/processing/test/holdout.csv"
OUT   = "/opt/ml/processing/output"

with tarfile.open(os.path.join(MODEL, "model.tar.gz")) as t:
    t.extractall("/tmp/model")
print("extracted:", os.listdir("/tmp/model"))

path = os.path.join("/tmp/model", "xgboost-model")
booster = xgb.Booster()
try:
    booster.load_model(path)            # native format, XGBoost >= 1.3
except Exception as e:
    print("load_model failed, trying pickle:", e)
    import pickle
    booster = pickle.load(open(path, "rb"))

df = pd.read_csv(TEST, header=None)
y  = df.iloc[:, 0].values
X  = df.iloc[:, 1:].values

pred   = booster.predict(xgb.DMatrix(X))
pr_auc = float(average_precision_score(y, pred))

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "evaluation.json"), "w") as f:
    json.dump({"metrics": {"pr_auc": {"value": pr_auc}}}, f, indent=2)

print("rows            :", len(y))
print("candidate PR-AUC:", round(pr_auc, 4))
