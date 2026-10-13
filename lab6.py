import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = next(p for p in ["ic272_lab6_vibration.csv", "Lab6_vibrations.csv"] if os.path.exists(p))
df = pd.read_csv(csv_path)
features = ["vib_x", "vib_y", "machine_hours"]
X = df[features].to_numpy(float)
y = df["faulty"].to_numpy()
idx = np.random.default_rng(0).permutation(len(y))
te, tr = idx[:100], idx[100:]
X_train, y_train = X[tr], y[tr]
X_test, y_test = X[te], y[te]
print("train:", X_train.shape, " test:", X_test.shape)


def pairwise_distances(A, B):
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def standardise(X_train, X_other):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    return (X_train - mu) / sigma, (X_other - mu) / sigma, mu, sigma


# Why this lab exists
fig, ax = plt.subplots(figsize=(5.5, 5.5))
for label, colour in [(0, "tab:blue"), (1, "tab:red")]:
    m = y == label
    ax.scatter(df.loc[m, "vib_x"], df.loc[m, "vib_y"], s=18, c=colour, alpha=0.7, label=f"faulty={label}")
ax.set_xlabel("vib_x"); ax.set_ylabel("vib_y"); ax.set_aspect("equal"); ax.legend()
plt.show()

baseline = max(y_test.mean(), 1 - y_test.mean())
print("baseline:", baseline)
for name, s in zip(features, X_train.std(axis=0)):
    print(f"{name:15s}{s:12.3f}")
sq = (X_test[:, None, :] - X_train[None, :, :]) ** 2
share = sq.sum(axis=1) / sq.sum(axis=(1, 2))[:, None]
for name, s in zip(features, share.mean(axis=0) * 100):
    print(f"{name:15s}{s:.7f} %")


# Task 1
def knn_predict(X_train, y_train, X_query, k):
    D = pairwise_distances(X_query, X_train)
    nn_idx = np.argsort(D, axis=1)[:, :k]
    return (y_train[nn_idx].mean(axis=1) > 0.5).astype(int)


pred_self = knn_predict(X_train, y_train, X_train, k=1)
print("k=1 on train rows matches labels:", np.array_equal(pred_self, y_train))
assert np.array_equal(pred_self, y_train)

# Task 2
Z_train, Z_test, mu, sigma = standardise(X_train, X_test)
acc_raw = np.mean(knn_predict(X_train, y_train, X_test, 7) == y_test)
acc_std = np.mean(knn_predict(Z_train, y_train, Z_test, 7) == y_test)
print(f"baseline {baseline:.2f} | raw {acc_raw:.2f} | standardised {acc_std:.2f}")


# Task 3
def k_fold_cv_knn(X, y, k, folds=5, seed=0):
    perm = np.random.default_rng(seed).permutation(len(y))
    parts = np.array_split(perm, folds)
    accs = []
    for i in range(folds):
        trn = np.concatenate([parts[j] for j in range(folds) if j != i])
        pred = knn_predict(X[trn], y[trn], X[parts[i]], k)
        accs.append(np.mean(pred == y[parts[i]]))
    return float(np.mean(accs))


ks = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 75]
cv_acc = np.array([k_fold_cv_knn(Z_train, y_train, k) for k in ks])
print(f"{'k':>4} | CV accuracy")
for k, a in zip(ks, cv_acc):
    print(f"{k:>4} | {a:.4f}")
best_k = ks[int(np.argmax(np.round(cv_acc, 10)))]
print("best k:", best_k)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(ks, cv_acc, "o-")
ax.axhline(baseline, color="gray", ls="--")
ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
ax.set_xlabel("k"); ax.set_ylabel("CV accuracy")
plt.show()

for k in sorted({1, best_k, 75}):
    print(f"k={k:>2}: test accuracy = {np.mean(knn_predict(Z_train, y_train, Z_test, k) == y_test):.2f}")


# H1
np.random.seed(0)
es = [0, 2, 5, 10, 20, 50]
h1_acc, h1_ratio = [], []
for e in es:
    A = np.hstack([X[:, :2], np.random.normal(0, 1, size=(len(y), e))]) if e > 0 else X[:, :2]
    Za_tr, Za_te, _, _ = standardise(A[tr], A[te])
    D = pairwise_distances(Za_te, Za_tr)
    h1_acc.append(np.mean(knn_predict(Za_tr, y[tr], Za_te, 7) == y[te]))
    h1_ratio.append(D.min() / D.max())
    print(f"e={e:>2}  acc={h1_acc[-1]:.2f}  min/max={h1_ratio[-1]:.3f}")

fig, ax1 = plt.subplots(figsize=(6.5, 4))
ax1.plot(es, h1_acc, "o-", color="tab:red"); ax1.set_xlabel("noise columns e"); ax1.set_ylabel("test accuracy", color="tab:red")
ax2 = ax1.twinx(); ax2.plot(es, h1_ratio, "s--", color="tab:blue"); ax2.set_ylabel("min(D)/max(D)", color="tab:blue")
plt.show()


# H2
def knn_predict_weighted(X_train, y_train, X_query, k, eps=1e-8):
    D = pairwise_distances(X_query, X_train)
    nn_idx = np.argsort(D, axis=1)[:, :k]
    w = 1.0 / (np.take_along_axis(D, nn_idx, axis=1) + eps)
    return (((w * y_train[nn_idx]).sum(axis=1) / w.sum(axis=1)) > 0.5).astype(int)


def k_fold_cv_generic(X, y, k, predict_fn, folds=5, seed=0):
    perm = np.random.default_rng(seed).permutation(len(y))
    parts = np.array_split(perm, folds)
    accs = []
    for i in range(folds):
        trn = np.concatenate([parts[j] for j in range(folds) if j != i])
        accs.append(np.mean(predict_fn(X[trn], y[trn], X[parts[i]], k) == y[parts[i]]))
    return float(np.mean(accs))


ks2 = ks + [101, 150]
cv_plain = [k_fold_cv_generic(Z_train, y_train, k, knn_predict) for k in ks2]
cv_wtd = [k_fold_cv_generic(Z_train, y_train, k, knn_predict_weighted) for k in ks2]
print(f"{'k':>4} | plain  | weighted")
for k, a, b in zip(ks2, cv_plain, cv_wtd):
    print(f"{k:>4} | {a:.4f} | {b:.4f}")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(ks2, cv_plain, "o-", label="plain")
ax.plot(ks2, cv_wtd, "s-", label="weighted")
ax.axhline(baseline, color="gray", ls="--")
ax.set_xscale("log"); ax.set_xticks(ks2); ax.set_xticklabels(ks2, rotation=45)
ax.set_xlabel("k"); ax.set_ylabel("CV accuracy"); ax.legend()
plt.show()


# H3
def confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)
    for t in (0, 1):
        for p in (0, 1):
            cm[t, p] = np.sum((y_true == t) & (y_pred == p))
    return cm


cm = confusion_matrix(y_test, knn_predict(Z_train, y_train, Z_test, best_k))
tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
precision, recall = tp / (tp + fp), tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print(pd.DataFrame(cm, index=["true 0", "true 1"], columns=["pred 0", "pred 1"]))
print(f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")


# H4
Z2_train, _, mu2, sigma2 = standardise(X_train[:, :2], X_test[:, :2])
gx = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 150)
gy = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 150)
GX, GY = np.meshgrid(gx, gy)
Zgrid = (np.column_stack([GX.ravel(), GY.ravel()]) - mu2) / sigma2

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for ax, k in zip(axes, [1, best_k]):
    region = knn_predict(Z2_train, y_train, Zgrid, k).reshape(GX.shape)
    ax.contourf(GX, GY, region, levels=[-0.5, 0.5, 1.5], colors=["#cfe0f5", "#f5cfcf"])
    for label, colour in [(0, "tab:blue"), (1, "tab:red")]:
        m = y_train == label
        ax.scatter(X_train[m, 0], X_train[m, 1], s=14, c=colour, edgecolors="k", linewidths=0.3)
    ax.set_title(f"k = {k}"); ax.set_xlabel("vib_x"); ax.set_ylabel("vib_y"); ax.set_aspect("equal")
plt.tight_layout()
plt.show()
