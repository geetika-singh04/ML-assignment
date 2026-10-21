import numpy as np, json, time, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

X, y = fetch_olivetti_faces(return_X_y=True)
s = np.load("split.npz"); tr, va, te = s["train"], s["val"], s["test"]
Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]
Xm = Xte.copy().reshape(-1, 64, 64); Xm[:, 24:40, 24:40] = 0; Xm = Xm.reshape(-1, 4096)
cfg = json.load(open("q1_config.json")); D, LEAF, BF = cfg["D"], cfg["LEAF"], cfg["BF"]
K = 40
acc = lambda p, t: np.mean(p == t)

# SAMME 
def samme_round(w, pred, y, K):
    """One SAMME weight update. Returns (error, alpha, new_w); alpha=None if learner is no better than random."""
    err = w[pred != y].sum() / w.sum()
    if err >= 1 - 1 / K:                     
        return err, None, w
    e = max(err, 1e-10)                       
    a = np.log((1 - e) / e) + np.log(K - 1)
    w = w * np.exp(a * (pred != y))
    return err, a, w / w.sum()

def fit_boost(X, y, depth, R):
    n = len(y); w = np.full(n, 1 / n)
    m = dict(H=[], A=[], err=[], W=[], P=[])
    for t in range(R):
        h = DecisionTreeClassifier(max_depth=depth, random_state=t).fit(X, y, sample_weight=w)
        p = h.predict(X)
        err, a, w_new = samme_round(w, p, y, K)
        if a is None:                         
            break
        m["H"].append(h); m["A"].append(a); m["err"].append(err); m["W"].append(w); m["P"].append(p)
        w = w_new
        if err == 0:                        
            break
    m["A"] = np.array(m["A"]); m["W"] = np.array(m["W"]); m["P"] = np.array(m["P"])
    return m

def boost_q(m, Xs, T):
    """Normalised weighted vote q_k(x) = S_k(x) / sum_t alpha_t using the first T learners."""
    S = np.zeros((len(Xs), K))
    for h, a in zip(m["H"][:T], m["A"][:T]):
        S[np.arange(len(Xs)), h.predict(Xs)] += a
    return S / m["A"][:T].sum()

def fit_bag(X, y, B, seed=0):
    r = np.random.default_rng(seed); out = []
    for b in range(B):
        i = r.integers(0, len(y), len(y))
        out.append(DecisionTreeClassifier(max_depth=D, min_samples_leaf=LEAF, random_state=b).fit(X[i], y[i]))
    return out

def bag_votes(trees, Xs):
    P = np.array([t.predict(Xs) for t in trees])           
    cnt = np.zeros((P.shape[1], K))
    for row in P:
        cnt[np.arange(P.shape[1]), row] += 1
    return cnt / len(trees), P

#  (a) 
np.set_printoptions(precision=4, suppress=True)
Xt = np.arange(1, 10, dtype=float).reshape(-1, 1)
yt = np.array([0, 0, 0, 1, 1, 0, 2, 2, 2])
w0 = np.full(9, 1 / 9)
h = DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xt, yt, sample_weight=w0)
pt = h.predict(Xt)
err, a, w1 = samme_round(w0, pt, yt, 3)
print("TOY SAMME round (K=3)")
print("true labels   :", yt)
print("initial w     :", w0)
print("weak preds    :", pt)
print(f"weighted error: {err:.4f}   alpha: {a:.4f}")
print("updated w     :", w1, " sum =", w1.sum())

#  (b) 
Rs = [10, 25, 50, 100]
models = {d: fit_boost(Xtr, ytr, d, 100) for d in [1, 2, 3, 4]}
best = None
print("\ndepth rounds(actual) train val")
for d, m in models.items():
    for R in Rs:
        T = min(R, len(m["A"]))
        tra = acc(boost_q(m, Xtr, T).argmax(1), ytr)
        v = acc(boost_q(m, Xva, T).argmax(1), yva)
        print(f"{d} {R}({T}) {tra:.3f} {v:.3f}")
        if best is None or v > best[0]:         
            best = (v, d, R, T)
_, DS, RS, TS = best
print(f"\nSELECTED: depth={DS}, rounds={RS} (actual {TS})")
M = models[DS]
json.dump({"DS": DS, "RS": RS}, open("q2_config.json", "w"))

W = M["W"][:TS]
fig, ax = plt.subplots(1, 4, figsize=(12, 2.6))
for a_, v_, t_ in zip(ax, [M["err"][:TS], M["A"][:TS], W.max(1), np.sort(W, 1)[:, -10:].sum(1)],
                      ["weighted train error", "learner weight $\\alpha_t$", "max sample weight", "weight of 10 heaviest"]):
    a_.plot(v_); a_.set_xlabel("round"); a_.set_title(t_, fontsize=9)
plt.tight_layout(); plt.savefig("q2_boost_curves.png", dpi=200)

#  (c) 
print("\nmethod  n  train val test time(s)")
for T in [25, 50, 100]:
    t0 = time.time(); mb = fit_boost(Xtr, ytr, DS, T); tb = time.time() - t0
    Tb = len(mb["A"])
    r = [acc(boost_q(mb, Z, Tb).argmax(1), t) for Z, t in [(Xtr, ytr), (Xva, yva), (Xte, yte)]]
    print(f"boost  {T:3d} " + " ".join(f"{v:.3f}" for v in r) + f" {tb:.2f}")
    t0 = time.time(); bt = fit_bag(Xtr, ytr, T); tg = time.time() - t0
    r = [acc(bag_votes(bt, Z)[0].argmax(1), t) for Z, t in [(Xtr, ytr), (Xva, yva), (Xte, yte)]]
    print(f"bag    {T:3d} " + " ".join(f"{v:.3f}" for v in r) + f" {tg:.2f}")

Wall, P = M["W"][:TS], M["P"][:TS]
i = np.argmax(Wall.max(0))
print(f"\nTrain face {i} (class {ytr[i]}): max weight {Wall[:, i].max():.4f} (uniform = {1/len(ytr):.4f}), "
      f"misclassified by {np.sum(P[:, i] != ytr[i])}/{TS} learners")
print("round weight pred correct   (first 15 rounds)")
for t in range(min(15, TS)):
    print(f"{t:3d}  {Wall[t, i]:.4f}  {P[t, i]:2d}  {P[t, i] == ytr[i]}")
plt.figure(figsize=(4, 2.6)); plt.plot(Wall[:, i])
wrong = np.where(P[:, i] != ytr[i])[0]
plt.scatter(wrong, Wall[wrong, i], c="r", marker="x", label="misclassified"); plt.legend(fontsize=7)
plt.xlabel("round"); plt.title(f"weight of train face {i}", fontsize=9); plt.tight_layout()
plt.savefig("q2_heavy_sample.png", dpi=200)

#  (d)
tree = DecisionTreeClassifier(max_depth=D, min_samples_leaf=LEAF, random_state=0).fit(Xtr, ytr)
bag = pickle.load(open("bag.pkl", "rb"))                       #
q = {}
for name, Z in [("clean", Xte), ("masked", Xm)]:
    qt = tree.predict_proba(Z); qb = bag_votes(bag, Z)[0]; qs = boost_q(M, Z, TS)
    q[name] = (qt, qb, qs)
    for mname, qq in zip(["tree", "bagging", "boosting"], [qt, qb, qs]):
        print(f"[{name}] {mname:8s} acc={acc(qq.argmax(1), yte):.3f}  mean winning prob/vote={qq.max(1).mean():.3f}")


bc, bm = q["clean"][1].argmax(1), q["masked"][1].argmax(1)
sc, sm = q["clean"][2].argmax(1), q["masked"][2].argmax(1)
cand = np.where(((bc == yte) != (bm == yte)) != ((sc == yte) != (sm == yte)))[0]
if len(cand):
    j = cand[0]
    cnt = q["masked"][1][j] * BF                               
    dis = 1 - (cnt * (cnt - 1)).sum() / (BF * (BF - 1))         
    cw = np.array([Wall.mean(0)[ytr == c].sum() for c in range(K)])  
    print(f"\nFace test idx {j}, true={yte[j]}")
    print(f"clean : bag={bc[j]} boost={sc[j]}   masked: bag={bm[j]} boost={sm[j]}")
    print(f"bag masked: winning vote={q['masked'][1][j].max():.3f}, tree disagreement={dis:.3f}")
    print(f"boost masked: winning q={q['masked'][2][j].max():.3f}, q(true)={q['masked'][2][j, yte[j]]:.3f}")
    print(f"avg boosting weight of class true={cw[yte[j]]:.3f}, class predicted by boost(masked)={cw[sm[j]]:.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(4, 2.2))
    for a_, Z, t_ in zip(ax, [Xte[j], Xm[j]], ["clean", "masked"]):
        a_.imshow(Z.reshape(64, 64), cmap="gray"); a_.axis("off"); a_.set_title(t_, fontsize=8)
    plt.tight_layout(); plt.savefig("q2_mask_case.png", dpi=200)
else:
    print("\nNo face where bagging and boosting differ in status; compare confidence instead.")
