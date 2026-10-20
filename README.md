# Bagging, Boosting and Bias–Variance on Olivetti Faces

**Name:** GEETIKA SINGH
**Roll number:** M26CPS003
**Repository:** https://github.com/geetika-singh04/ML-assignment
**Final commit hash:** 8f99a2a5980d9ece15ad9477cd017be07a42fc6f

Machine Learning Assignment 2. Decision tree baseline, bagging and SAMME boosting (both implemented from scratch),
and measured bias/variance, on `fetch_olivetti_faces` (40 identities, 10 images each, 64x64, raw flattened pixels,
no PCA or other dimensionality reduction).

## Setup

- Python 3.12
- Install packages (needs internet the first time: scikit-learn downloads the Olivetti data to `~/scikit_learn_data`):

```
pip install -r requirements.txt
```

## Seed

The seed is formed from **all digits of the roll number, in order**. Set it once at the top of `q1.py`:

```python
ROLL = "YOUR_ROLL_NUMBER"
```

`q1.py` creates the split and saves the seed to `q1_config.json`. `q2.py` and `q3.py` read the split and the config,
so they never need the roll number again.

## Exact reproduction commands

Run from the repository root, **in this order** (each script needs files made by the previous one).

Windows (cmd):

```
python q1.py 
python q2.py 
python q3.py 
python extras.py 
```

Linux / macOS / PowerShell can use `python q1.py ` (and likewise for the others).

Runtime: `q1.py` takes a few minutes, `q2.py` and `q3.py` take much longer (they train several thousand trees on
4096-dimensional inputs). Run them in a normal terminal, not under a debugger.

## What each file does

| File | Purpose | Produces |
|---|---|---|
| `q1.py` | Split (6 train / 2 val / 2 test per identity), single-tree grid, bagging from scratch, OOB / disagreement / margin, tree vs bagging on clean and masked test sets | `split.npz`, `q1_config.json`, `bag.pkl`, `q1_curves.png`, `cases.png` |
| `q2.py` | SAMME from scratch (toy round, depth/rounds selection, boosting vs bagging, heavy-weight face, robustness) | `q2_config.json`, `q2_boost_curves.png`, `q2_heavy_sample.png`, `q2_mask_case.png` |
| `q3.py` | 30 within-identity resamples, measured Bias² and Variance for tree / bagging / boosting, depth study | `resamples.npy`, `q3_depth.png` |
| `extras.py` | Leaf statistics per depth, share of splits inside the masked block, prediction changes under the mask (numbers used in the report) | printed to `extras_log.txt` |
| `report.tex` | Report source (CVPR template) | PDF |

All figures are produced inside `q1.py`, `q2.py` and `q3.py`.

## Saved artefacts (committed)

- `split.npz` – the train/validation/test indices used by every experiment
- `q1_config.json` – frozen tree setting (`D`, `LEAF`), final bagging size `BF`, and `SEED`
- `q2_config.json` – selected boosting depth and rounds (`DS`, `RS`)
- `bag.pkl` – the final bagged trees from Q1 (reused in Q2 and Q3)
- `resamples.npy` – the 30 resampled training sets used in Q3
- `q1_log.txt`, `q2_log.txt`, `q3_log.txt`, `extras_log.txt` – final logs

## Protocol summary

- Raw flattened pixels only. Models are trained on clean images only.
- Masked test set: rows and columns 24 to 39 (centred 16x16 block) set to zero.
- All selection (tree setting, bagging size, boosting depth and rounds) uses validation accuracy only.
  Ties go to the first setting in the grid.
- Bagging and SAMME are implemented in this repository; only `DecisionTreeClassifier` comes from scikit-learn.
- SAMME: `alpha_t = log((1-eps_t)/eps_t) + log(K-1)`. A weak learner with weighted error ≥ `1-1/K` is rejected and boosting stops.
  Confidence is the normalised weighted vote `q_k(x) = S_k(x) / sum_t alpha_t`.
- Bias² and Variance use the class vectors specified in the assignment (tree: `predict_proba`; bagging: vote fractions;
  boosting: normalised weighted vote) over 30 resampled training sets, with the clean test set fixed.
