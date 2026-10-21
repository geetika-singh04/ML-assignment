import numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier


K = 40
acc = lambda p, t: np.mean(p == t)
X, y = fetch_olivetti_faces(return_X_y=True)
s = np.load("split.npz"); tr, te = s["train"], s["test"]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
Xm = Xte.copy().reshape(-1, 64, 64); Xm[:, 24:40, 24:40] = 0; Xm = Xm.reshape(-1, 4096)

def samme_round(w, pred, y, K):
    err = w[pred != y].sum() / w.sum()
    if err >= 1 - 1 / K:
        return err, None, w
    e = max(err, 1e-10)
    a = np.log((1 - e) / e) + np.log(K - 1)
    w = w * np.exp(a * (pred != y))
    return err, a, w / w.sum()

def fit_boost(X, y, depth, R):
    n = len(y); w = np.full(n, 1 / n); H, A = [], []
    for t in range(R):
        h = DecisionTreeClassifier(max_depth=depth, random_state=t).fit(X, y, sample_weight=w)
        err, a, w_new = samme_round(w, h.predict(X), y, K)
        if a is None:
            break
        H.append(h); A.append(a); w = w_new
        if err == 0:
            break
    return H, np.array(A)

def boost_q(m, Xs):
    H, A = m; S = np.zeros((len(Xs), K))
    for h, a in zip(H, A):
        S[np.arange(len(Xs)), h.predict(Xs)] += a
    return S / A.sum()

def fit_bag(X, y, B, depth, leaf, seed=0):
    r = np.random.default_rng(seed); out = []
    for b in range(B):
        i = r.integers(0, len(y), len(y))
        out.append(DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=b).fit(X[i], y[i]))
    return out

def bag_votes(trees, Xs):
    P = np.array([t.predict(Xs) for t in trees]); cnt = np.zeros((P.shape[1], K))
    for row in P:
        cnt[np.arange(P.shape[1]), row] += 1
    return cnt / len(trees), P

c1, c2 = json.load(open("q1_config.json")), json.load(open("q2_config.json"))
D, LEAF, BF, SEED, DS, RS = c1["D"], c1["LEAF"], c1["BF"], c1["SEED"], c2["DS"], c2["RS"]
R = 30
E = np.eye(K)[yte]                                   

rng = np.random.default_rng(SEED)
resamples = np.array([np.concatenate([rng.choice(np.where(ytr == c)[0], 6, replace=True) for c in range(K)])
                      for _ in range(R)])
np.save("resamples.npy", resamples)

def bias_var(Pr):                                    
    pbar = Pr.mean(0)
    return ((pbar - E) ** 2).sum(1).mean(), ((Pr - pbar) ** 2).sum(2).mean()

def tree_fit(Xr, yr, depth):
    return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=LEAF, random_state=0).fit(Xr, yr)

#(a)
out = {n: {"clean": [], "masked": []} for n in ["tree", "bagging", "boosting"]}
for r in range(R):
    Xr, yr = Xtr[resamples[r]], ytr[resamples[r]]
    t = tree_fit(Xr, yr, D)
    bag = fit_bag(Xr, yr, BF, D, LEAF, seed=r)
    bst = fit_boost(Xr, yr, DS, RS)
    for k, Z in [("clean", Xte), ("masked", Xm)]:
        out["tree"][k].append(t.predict_proba(Z))
        out["bagging"][k].append(bag_votes(bag, Z)[0])
        out["boosting"][k].append(boost_q(bst, Z))
    print("refit", r + 1, "/", R, flush=True)

table = {}
print("\n(a) model | clean acc mean+-std | masked acc mean | bias2 | variance | masked variance")
for n, d in out.items():
    Pc, Pm = np.array(d["clean"]), np.array(d["masked"])
    ac = [acc(p.argmax(1), yte) for p in Pc]; am = [acc(p.argmax(1), yte) for p in Pm]
    b2, v = bias_var(Pc); _, vm = bias_var(Pm)
    table[n] = (np.mean(ac), np.mean(am), b2, v)
    print(f"{n:9s} {np.mean(ac):.3f}+-{np.std(ac):.3f}  {np.mean(am):.3f}  {b2:.4f}  {v:.4f}  {vm:.4f}")

#  (b)
depths = list(dict.fromkeys([2, D, None]))           
if len(depths) < 3:                                   
    depths = [2, 8, None]
rows = []
print("\n(b) depth | tree: acc bias2 var leaf_occ frac1 | bag100: acc bias2 var disagree | var removed, acc gain")
for d in depths:
    Pt, Pb, occ, one, dis = [], [], [], [], []
    for r in range(R):
        Xr, yr = Xtr[resamples[r]], ytr[resamples[r]]
        t = tree_fit(Xr, yr, d); T = t.tree_; leaf = T.children_left == -1
        occ.append(len(yr) / leaf.sum()); one.append((T.n_node_samples[leaf] == 1).mean())
        Pt.append(t.predict_proba(Xte))
        v, P = bag_votes(fit_bag(Xr, yr, 100, d, LEAF, seed=r), Xte)
        Pb.append(v)
        cnt = v * 100; dis.append((1 - (cnt * (cnt - 1)).sum(1) / (100 * 99)).mean())   
    Pt, Pb = np.array(Pt), np.array(Pb)
    at = np.mean([acc(p.argmax(1), yte) for p in Pt]); ab = np.mean([acc(p.argmax(1), yte) for p in Pb])
    (bt, vt), (bb, vb) = bias_var(Pt), bias_var(Pb)
    rows.append((str(d), bt, vt, bb, vb))
    print(f"{str(d):5s} | {at:.3f} {bt:.4f} {vt:.4f} {np.mean(occ):.2f} {np.mean(one):.3f} | "
          f"{ab:.3f} {bb:.4f} {vb:.4f} {np.mean(dis):.3f} | {vt - vb:.4f} ({(vt - vb) / vt:.0%}), {ab - at:+.3f}")

lab = [r[0] for r in rows]
plt.figure(figsize=(4.5, 3))
for j, (nm, st) in enumerate([("tree bias$^2$", "o-"), ("tree var", "o--"), ("bag bias$^2$", "s-"), ("bag var", "s--")]):
    plt.plot(lab, [r[j + 1] for r in rows], st, label=nm)
plt.xlabel("max depth"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig("q3_depth.png", dpi=200)

#  (d) 
print("\n(d) model | clean acc | masked acc | bias2 | variance")
for n, (ac, am, b2, v) in table.items():
    print(f"{n:9s} {ac:.3f}  {am:.3f}  {b2:.4f}  {v:.4f}")
