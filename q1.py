import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

# ---------------- setup ----------------
ROLL = "M26CPS003"          
SEED = int("".join(c for c in ROLL if c.isdigit()))
rng = np.random.default_rng(SEED)

data = fetch_olivetti_faces()
X, y = data.data, data.target       # X: (400, 4096) raw flattened pixels

# per-identity split: 6 train / 2 val / 2 test, saved once
tr, va, te = [], [], []
for c in range(40):
    idx = rng.permutation(np.where(y == c)[0])
    tr += list(idx[:6]); va += list(idx[6:8]); te += list(idx[8:])
tr, va, te = map(np.array, (tr, va, te))
np.savez("split.npz", train=tr, val=va, test=te)

Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]

# masked test set: zero rows/cols 24..39
Xte_m = Xte.copy().reshape(-1, 64, 64)
Xte_m[:, 24:40, 24:40] = 0
Xte_m = Xte_m.reshape(-1, 4096)

acc = lambda p, t: np.mean(p == t)

# ---------------- (a) single tree ----------------
best = None
for d in [2, 4, 8, 16, None]:
    for leaf in [1, 2, 4]:
        t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=leaf, random_state=0).fit(Xtr, ytr)
        v = acc(t.predict(Xva), yva)
        print(f"depth={d} leaf={leaf} train={acc(t.predict(Xtr), ytr):.3f} val={v:.3f}")
        if best is None or v > best[0]:      # ties -> first in grid
            best = (v, d, leaf)
_, D, LEAF = best
print("\nFROZEN:", "max_depth =", D, "min_samples_leaf =", LEAF)

tree = DecisionTreeClassifier(max_depth=D, min_samples_leaf=LEAF, random_state=0).fit(Xtr, ytr)
T = tree.tree_
is_leaf = T.children_left == -1
n_leaves = is_leaf.sum()
print(f"train={acc(tree.predict(Xtr), ytr):.3f} val={acc(tree.predict(Xva), yva):.3f} "
      f"test={acc(tree.predict(Xte), yte):.3f}")
print(f"leaves={n_leaves} mean samples/leaf={len(ytr)/n_leaves:.2f} "
      f"frac single-example leaves={(T.n_node_samples[is_leaf] == 1).mean():.3f}")

# ---------------- (b) bagging from scratch ----------------
n = len(ytr)
Bs = [1, 5, 10, 25, 50, 100, 200]
trees, in_bag, distinct = [], [], []
for b in range(max(Bs)):
    boot = rng.integers(0, n, n)                       # bootstrap of size n
    m = DecisionTreeClassifier(max_depth=D, min_samples_leaf=LEAF, random_state=b)
    trees.append(m.fit(Xtr[boot], ytr[boot]))
    mask = np.zeros(n, bool); mask[boot] = True
    in_bag.append(mask); distinct.append(mask.sum())
in_bag = np.array(in_bag)

def tree_preds(B, Xs):                                 # (B, n_samples) labels
    return np.array([trees[i].predict(Xs) for i in range(B)])

def votes(P):                                          # (n_samples, 40) vote counts
    cnt = np.zeros((P.shape[1], 40), int)
    for row in P:
        cnt[np.arange(P.shape[1]), row] += 1
    return cnt

expected = n * (1 - (1 - 1 / n) ** n)
print(f"\nTheoretical expected distinct per bootstrap = {expected:.2f}")

# ---------------- (b)+(c) metrics for each B ----------------
res = {k: [] for k in ["train", "val", "test", "dist", "oob", "dis", "margin"]}
for B in Bs:
    Ptr, Pva, Pte = tree_preds(B, Xtr), tree_preds(B, Xva), tree_preds(B, Xte)
    res["train"].append(acc(votes(Ptr).argmax(1), ytr))
    res["val"].append(acc(votes(Pva).argmax(1), yva))
    res["test"].append(acc(votes(Pte).argmax(1), yte))
    res["dist"].append(np.mean(distinct[:B]))

    # OOB: only trees that did not see the sample
    cnt = np.zeros((n, 40), int)
    for i in range(B):
        oob = ~in_bag[i]
        cnt[np.where(oob)[0], Ptr[i][oob]] += 1
    has = cnt.sum(1) > 0
    res["oob"].append(acc(cnt[has].argmax(1), ytr[has]))

    # pairwise disagreement and vote margin on validation set
    if B > 1:
        d = [(Pva[i + 1:] != Pva[i]).mean(1).sum() for i in range(B - 1)]
        res["dis"].append(np.sum(d) / (B * (B - 1) / 2))
    else:
        res["dis"].append(np.nan)
    frac = np.sort(votes(Pva) / B, axis=1)
    res["margin"].append(np.mean(frac[:, -1] - frac[:, -2]))

print("\n  B | train  val   test | distinct |  OOB   disagree margin")
for i, B in enumerate(Bs):
    print(f"{B:4d} | {res['train'][i]:.3f} {res['val'][i]:.3f} {res['test'][i]:.3f} | "
          f"{res['dist'][i]:7.1f}  | {res['oob'][i]:.3f}  {res['dis'][i]:.3f}   {res['margin'][i]:.3f}")

# plot (c)
fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
for a, k, t in zip(ax, ["oob", "dis", "margin"], ["OOB accuracy", "Pairwise disagreement", "Mean vote margin"]):
    a.plot(Bs, res[k], "o-"); a.set_xscale("log"); a.set_xlabel("B"); a.set_title(t)
plt.tight_layout(); plt.savefig("q1_curves.png", dpi=200)

# final B: best validation accuracy, ties -> smallest B
BF = Bs[int(np.argmax(res["val"]))]
print("\nFINAL B =", BF)

# save frozen choices for Q2 / Q3
import json, pickle
json.dump({"D": D, "LEAF": LEAF, "BF": BF, "SEED": SEED}, open("q1_config.json", "w"))
pickle.dump(trees[:BF], open("bag.pkl", "wb"))

# ---------------- (d) tree vs bagging, clean vs masked ----------------
out = {}
for name, Xs in [("clean", Xte), ("masked", Xte_m)]:
    tp = tree.predict(Xs)
    tconf = tree.predict_proba(Xs).max(1)
    cnt = votes(tree_preds(BF, Xs))
    bp = cnt.argmax(1)
    bconf = cnt.max(1) / BF
    out[name] = (tp, bp, cnt)
    print(f"\n[{name}] tree acc={acc(tp, yte):.3f} conf={tconf.mean():.3f} | "
          f"bag acc={acc(bp, yte):.3f} conf={bconf.mean():.3f}")

# example faces (clean test set)
tp, bp, cnt = out["clean"]
fixed = np.where((tp != yte) & (bp == yte))[0][:2]
both_wrong = np.where((tp != yte) & (bp != yte))[0][:2]
cases = [("bag corrects tree", i) for i in fixed] + [("both wrong", i) for i in both_wrong]

fig, ax = plt.subplots(1, len(cases), figsize=(3 * len(cases), 3.3))
ax = np.atleast_1d(ax)
for a, (kind, i) in zip(ax, cases):
    print(f"\n{kind}: test idx {i}, true={yte[i]}, tree pred={tp[i]}, bag pred={bp[i]}, "
          f"votes for true={cnt[i, yte[i]]}, votes for bag pred={cnt[i, bp[i]]} (of {BF})")
    a.imshow(Xte[i].reshape(64, 64), cmap="gray"); a.axis("off")
    a.set_title(f"{kind}\ntrue {yte[i]} | tree {tp[i]} | bag {bp[i]}\nvotes: true {cnt[i, yte[i]]}, pred {cnt[i, bp[i]]}",
                fontsize=7)
plt.tight_layout(); plt.savefig("cases.png", dpi=200)
