import numpy as np, json, pickle
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

K = 40
X, y = fetch_olivetti_faces(return_X_y=True)
s = np.load("split.npz"); tr, te = s["train"], s["test"]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
Xm = Xte.copy().reshape(-1, 64, 64); Xm[:, 24:40, 24:40] = 0; Xm = Xm.reshape(-1, 4096)
c1, c2 = json.load(open("q1_config.json")), json.load(open("q2_config.json"))

# ---- 1. Table 1: leaf statistics for every depth (leaf=1) ----
print("TABLE 1")
for d in [2, 4, 8, 16, None]:
    t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=1, random_state=0).fit(Xtr, ytr)
    T = t.tree_; L = T.children_left == -1
    print(f"depth={d}: leaves={L.sum()} samples/leaf={len(ytr)/L.sum():.2f} "
          f"single-example={(T.n_node_samples[L] == 1).mean():.3f} actual_depth={t.get_depth()}")

# ---- 2. Do the models split on pixels inside the masked block? ----
def in_block(trees):
    inside = total = 0
    for t in trees:
        f = t.tree_.feature; f = f[f >= 0]                    # internal nodes only
        r, c = f // 64, f % 64
        inside += ((r >= 24) & (r < 40) & (c >= 24) & (c < 40)).sum(); total += len(f)
    return inside, total

def fit_boost(X, y, depth, R):                                  # same as q2.py (deterministic)
    n = len(y); w = np.full(n, 1 / n); H, A = [], []
    for t in range(R):
        h = DecisionTreeClassifier(max_depth=depth, random_state=t).fit(X, y, sample_weight=w)
        p = h.predict(X); err = w[p != y].sum() / w.sum()
        if err >= 1 - 1 / K: break
        e = max(err, 1e-10); a = np.log((1 - e) / e) + np.log(K - 1)
        w = w * np.exp(a * (p != y)); w /= w.sum(); H.append(h); A.append(a)
        if err == 0: break
    return H, np.array(A)

def boost_pred(m, Z):
    H, A = m; S = np.zeros((len(Z), K))
    for h, a in zip(H, A): S[np.arange(len(Z)), h.predict(Z)] += a
    return S.argmax(1)

tree = DecisionTreeClassifier(max_depth=c1["D"], min_samples_leaf=c1["LEAF"], random_state=0).fit(Xtr, ytr)
bag = pickle.load(open("bag.pkl", "rb"))
boost = fit_boost(Xtr, ytr, c2["DS"], c2["RS"])

print("\nSPLITS INSIDE THE MASKED BLOCK (the block is 256/4096 = 6.25% of the pixels)")
for name, trees in [("single tree", [tree]), ("bagging", bag), ("boosting", boost[0])]:
    i, n = in_block(trees)
    print(f"{name:12s}: {i}/{n} splits = {i/n:.3f}")

print("\nPREDICTION CHANGES CLEAN -> MASKED (80 test faces)")
preds = {"single tree": (tree.predict(Xte), tree.predict(Xm)),
         "bagging": (np.array([t.predict(Xte) for t in bag]), np.array([t.predict(Xm) for t in bag])),
         "boosting": (boost_pred(boost, Xte), boost_pred(boost, Xm))}
for name, (pc, pm) in preds.items():
    if name == "bagging":                                       # plurality vote
        vote = lambda P: np.array([np.bincount(P[:, i], minlength=K).argmax() for i in range(P.shape[1])])
        pc, pm = vote(pc), vote(pm)
    print(f"{name:12s}: changed={np.sum(pc != pm)}  correct->wrong={np.sum((pc == yte) & (pm != yte))}  "
          f"wrong->correct={np.sum((pc != yte) & (pm == yte))}")
