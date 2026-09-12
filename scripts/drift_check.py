import json, os
import numpy as np, pandas as pd

IN_W   = "/opt/ml/processing/input/window/window.csv"
IN_REF = "/opt/ml/processing/input/reference/reference.json"
OUT    = "/opt/ml/processing/output"

def psi_score(ref, X, eps=1e-6):
    out = {}
    for c, r in ref.items():
        if c not in X.columns:
            continue
        s = X[c]
        if r["type"] == "cat":
            s = s.round().astype("Int64").astype(str)
            cur = s.value_counts(normalize=True)
            levels = [str(l) for l in r["levels"]]
            a = [0.0 if l == "__other__" else float(cur.get(l, 0.0)) for l in levels]
            if "__other__" in levels:
                i = levels.index("__other__")
                a[i] = max(0.0, 1.0 - sum(x for j, x in enumerate(a) if j != i))
            a = np.array(a)
        else:
            counts = np.histogram(s.dropna(), bins=np.array(r["edges"]))[0]
            a = counts / max(counts.sum(), 1)
        e = np.array(r["p"])
        e, a = np.clip(e, eps, None), np.clip(a, eps, None)
        e, a = e / e.sum(), a / a.sum()
        out[c] = float(np.sum((a - e) * np.log(a / e)))
    return out

X   = pd.read_csv(IN_W)
ref = json.load(open(IN_REF))
per = psi_score(ref, X)
mean_psi = float(np.mean(list(per.values())))

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "drift.json"), "w") as f:
    json.dump({"drift": {"mean_psi": {"value": mean_psi}},
               "per_feature": per}, f, indent=2)

print("features scored :", len(per))
print("mean PSI        :", round(mean_psi, 4))
