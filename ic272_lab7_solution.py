# IC-272 Lab 7: Logistic Regression
# Only numpy, pandas and matplotlib are used. No scikit-learn.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# Helper functions from earlier labs
# (design matrix + normal equation: Lab 4, standardise: Lab 5,
#  confusion matrix + precision/recall/F1: Lab 1)
# ---------------------------------------------------------------
def design_matrix(X):
    # put a column of ones in front so theta[0] is the intercept
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(len(X)), X])


def fit_normal_equation(X, y):
    # closed-form linear regression: solve (A^T A) theta = A^T y
    A = design_matrix(X)
    return np.linalg.solve(A.T @ A, A.T @ y)


def standardise(X_train, X_test):
    # mean and std come from the TRAINING set only, and are reused on the test set
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


def confusion_matrix(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_recall_f1(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


# ---------------------------------------------------------------
# Setting up
# ---------------------------------------------------------------
csv_name = "ic272_lab7_loans.csv"
if not os.path.exists(csv_name):
    csv_name = "lab7_loans.csv"   # my copy of the file has the shorter name

df = pd.read_csv(csv_name)
features = ["income_k", "utilisation"]
X = df[features].to_numpy(float)
y = df["defaulted"].to_numpy(int)

idx = np.random.default_rng(0).permutation(len(y))
te, tr = idx[:100], idx[100:]
X_train, y_train = X[tr], y[tr]
X_test, y_test = X[te], y[te]
print("train:", X_train.shape, " test:", X_test.shape)   # (400, 2) (100, 2)

print("baseline accuracy (predict 'repaid' for everyone):", 1 - y_test.mean())


# ---------------------------------------------------------------
# Why not just use linear regression?
# ---------------------------------------------------------------
theta_lin = fit_normal_equation(X_train, y_train.astype(float))
pred = design_matrix(X) @ theta_lin
print("smallest prediction:", pred.min().round(3))
print("largest prediction: ", pred.max().round(3))
print("predictions below 0:", (pred < 0).sum())
# a straight line goes below 0, which can't be a probability -> use the sigmoid


# ===============================================================
# TASK 1: Sigmoid and log-likelihood
# ===============================================================

# Task 1 (a): vectorised sigmoid
def sigmoid(z):
    z = np.clip(z, -500, 500)   # avoids overflow in exp() for huge |z|
    return 1.0 / (1.0 + np.exp(-z))


# Task 1 (b): log-likelihood. X has no ones column, A is built inside.
def log_likelihood(X, y, theta):
    A = design_matrix(X)
    h = sigmoid(A @ theta)
    eps = 1e-12   # tiny constant so log(0) never happens
    return np.sum(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))


# Task 1 (c): standardise, then evaluate at theta = 0
X_train_s, X_test_s, mu, sigma = standardise(X_train, X_test)

theta0 = np.zeros(X_train_s.shape[1] + 1)
total = log_likelihood(X_train_s, y_train, theta0)
print("total log-likelihood at theta=0:", round(total, 4))
print("per example:", round(total / len(y_train), 4))

# hand check: at theta = 0 every prediction is sigmoid(0) = 0.5, so each example
# contributes log(0.5) whether y is 0 or 1. Total = 400 * log(0.5) = -277.26
print("hand calculation, log(0.5):", round(np.log(0.5), 4))


# ===============================================================
# TASK 2: Fitting by gradient ascent
# ===============================================================

# Task 2 (a): same loop as Lab 4 gradient descent, with two changes
def fit_logistic(X, y, lr=0.1, n_iters=2000):
    A = design_matrix(X)
    m = len(y)
    theta = np.zeros(A.shape[1])
    history = []
    for _ in range(n_iters):
        h = sigmoid(A @ theta)                       # change 1: prediction goes through sigmoid
        theta = theta + (lr / m) * (A.T @ (y - h))   # change 2: PLUS sign, we are maximising
        history.append(log_likelihood(X, y, theta))
    return theta, history


# Task 2 (b): fit on the standardised training set and print theta
theta, ll_history = fit_logistic(X_train_s, y_train, lr=0.1, n_iters=2000)
print("theta (intercept, income, utilisation):", theta.round(4))

# the curve should never go down; if it does the sign is wrong
print("any decrease in log-likelihood?", bool((np.diff(ll_history) < -1e-9).any()))

# Task 2 (c): log-likelihood vs iteration
plt.figure(figsize=(6, 4))
plt.plot(ll_history)
plt.xlabel("iteration")
plt.ylabel("log-likelihood")
plt.title("Task 2: gradient ascent on the training set")
plt.grid(alpha=0.3)
plt.show()


# ===============================================================
# TASK 3: Thresholds and evaluation
# ===============================================================

# Task 3 (a): predicted probability of default for every test applicant
# (X_test_s already uses the training mu and sigma)
p_test = sigmoid(design_matrix(X_test_s) @ theta)
print("first five probabilities:", p_test[:5].round(3))


def evaluate(y_true, p, threshold):
    y_pred = (p >= threshold).astype(int)
    acc = accuracy(y_true, y_pred)
    prec, rec, f1 = precision_recall_f1(y_true, y_pred)
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    print(f"threshold {threshold}:  accuracy={acc:.3f}  precision={prec:.3f}  "
          f"recall={rec:.3f}  F1={f1:.3f}   (TP={tp}, FP={fp}, FN={fn}, TN={tn})")


print("baseline accuracy:", round(1 - y_test.mean(), 3))

# Task 3 (b): threshold 0.5
evaluate(y_test, p_test, 0.5)

# Task 3 (c): threshold 0.3
evaluate(y_test, p_test, 0.3)

# Viva notes:
# Lowering the threshold from 0.5 to 0.3 flags more people as defaulters, so recall
# rises sharply (fewer missed defaulters) while precision falls (more false alarms).
# As the bank I would use 0.3: a missed default costs the whole loan, while wrongly
# refusing a good customer only costs some interest. If refusing good customers were
# the bigger cost, a higher threshold would be better.


# ===============================================================
# HOMEWORK
# ===============================================================

# H1: ROC curve and AUC (names fixed by the lab, later labs call them)
def roc_curve_scratch(y_true, p):
    # use every predicted probability as a threshold, highest first
    thresholds = np.sort(np.unique(p))[::-1]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    fpr = [0.0]
    tpr = [0.0]
    for t in thresholds:
        y_pred = (p >= t).astype(int)
        tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
        tpr.append(tp / n_pos)   # recall
        fpr.append(fp / n_neg)
    return np.array(fpr), np.array(tpr)


def auc_trapezoid(fpr, tpr):
    order = np.argsort(fpr, kind="stable")   # increasing FPR
    fpr, tpr = fpr[order], tpr[order]
    area = 0.0
    for i in range(len(fpr) - 1):
        area += (fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2
    return area


fpr, tpr = roc_curve_scratch(y_test, p_test)
auc = auc_trapezoid(fpr, tpr)
print("AUC:", round(auc, 4))

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"logistic regression (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="random guessing")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("H1: ROC curve on the test set")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# AUC = 0.5: the model ranks defaulters no better than a coin flip (the diagonal).
# AUC = 1.0: every defaulter gets a higher probability than every non-defaulter,
# so some threshold separates the two classes perfectly.


# H2: decision boundary
# boundary: theta0 + theta1*(inc - mu0)/s0 + theta2*(util - mu1)/s1 = 0
# solving for utilisation undoes the standardisation
inc_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
util_line = mu[1] - sigma[1] * (theta[0] + theta[1] * (inc_line - mu[0]) / sigma[0]) / theta[2]

plt.figure(figsize=(7, 5))
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="tab:blue", s=15, alpha=0.6, label="repaid")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:red", s=15, alpha=0.6, label="defaulted")
plt.plot(inc_line, util_line, "k-", linewidth=2, label="boundary (p = 0.5)")
plt.ylim(X[:, 1].min() - 0.05, X[:, 1].max() + 0.05)
plt.xlabel("income_k")
plt.ylabel("utilisation")
plt.legend()
plt.title("H2: decision boundary of the loan model")
plt.show()


# H3: where logistic regression fails (Lab 6 ring data)
# Needs ic272_lab6_vibration.csv next to this file, otherwise it is skipped.
def knn_predict(X_tr, y_tr, X_te, k=5):
    # quick stand-in for the Lab 6 k-NN, swap in your own if you want the exact number
    preds = []
    for point in X_te:
        dists = np.sqrt(((X_tr - point) ** 2).sum(axis=1))
        nearest = np.argsort(dists)[:k]
        preds.append(int(y_tr[nearest].mean() >= 0.5))
    return np.array(preds)


ring_file = "ic272_lab6_vibration.csv"
if os.path.exists(ring_file):
    ring = pd.read_csv(ring_file)
    label_col = [c for c in ring.columns if c not in ("vib_x", "vib_y")][-1]
    Xr = ring[["vib_x", "vib_y"]].to_numpy(float)
    yr = ring[label_col].to_numpy(int)

    # change this to the exact Lab 6 split if it was different
    ridx = np.random.default_rng(0).permutation(len(yr))
    n_te = len(yr) // 5
    r_te, r_tr = ridx[:n_te], ridx[n_te:]

    # logistic on (vib_x, vib_y) only
    Xr_tr_s, Xr_te_s, _, _ = standardise(Xr[r_tr], Xr[r_te])
    th_r, _ = fit_logistic(Xr_tr_s, yr[r_tr])
    pred_r = (sigmoid(design_matrix(Xr_te_s) @ th_r) >= 0.5).astype(int)
    print("logistic (vib_x, vib_y) accuracy:", accuracy(yr[r_te], pred_r))
    print("k-NN accuracy:", accuracy(yr[r_te], knn_predict(Xr[r_tr], yr[r_tr], Xr[r_te])))

    # add the extra feature r^2 = vib_x^2 + vib_y^2 and refit
    r2 = (Xr ** 2).sum(axis=1, keepdims=True)
    Xr2 = np.hstack([Xr, r2])
    Xr2_tr_s, Xr2_te_s, _, _ = standardise(Xr2[r_tr], Xr2[r_te])
    th_r2, _ = fit_logistic(Xr2_tr_s, yr[r_tr])
    pred_r2 = (sigmoid(design_matrix(Xr2_te_s) @ th_r2) >= 0.5).astype(int)
    print("logistic with r^2 accuracy:", accuracy(yr[r_te], pred_r2))
else:
    print("Lab 6 ring data not found, skipping H3.")
# Why accuracy jumps: one class is a ring around the other, so no straight line in
# (x, y) separates them. The class depends on distance from the centre, i.e. r^2.
# With r^2 as a feature the boundary is linear in (x, y, r^2), which is a circle
# when drawn back in (x, y).


# H4: why standardise? Raw features with two learning rates
_, ll_small = fit_logistic(X_train, y_train, lr=0.001, n_iters=2000)
_, ll_big = fit_logistic(X_train, y_train, lr=0.01, n_iters=2000)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(ll_small)
axes[0].set_title("H4: raw features, lr = 0.001")
axes[1].plot(ll_big)
axes[1].set_title("H4: raw features, lr = 0.01")
for ax in axes:
    ax.set_xlabel("iteration")
    ax.set_ylabel("log-likelihood")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("lr=0.001: first / last log-likelihood:", round(ll_small[0], 1), round(ll_small[-1], 1))
print("lr=0.01 : first / last log-likelihood:", round(ll_big[0], 1), round(ll_big[-1], 1))
print("lr=0.01 : any decrease?", bool((np.diff(ll_big) < -1e-9).any()))
# lr = 0.001: income is in the hundreds, so its gradient dwarfs the others and the
#   learning rate must be tiny to stay stable. The curve crawls upwards very slowly.
# lr = 0.01: too big for the income direction, the weights overshoot, the
#   log-likelihood swings and ends up worse than the start.
# Same story as Lab 4: unscaled features give a long thin valley and one learning
# rate can't suit both directions. Standardising makes it round, so lr = 0.1 works.
