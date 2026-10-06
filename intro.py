# ================= IC-272 Lab 5: Standardisation and Ridge Regression =================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# ---------- Setup (exactly as given in the handout) ----------
df = pd.read_csv("ic272_lab5_rent.csv")
features = [c for c in df.columns if c != "rent"]
X = df[features].to_numpy(float)
y = df["rent"].to_numpy(float)
 
idx = np.random.default_rng(0).permutation(len(y))
tr, te = idx[:40], idx[40:]
X_train, X_test = X[tr], X[te]
y_train, y_test = y[tr], y[te]
print("train:", X_train.shape, " test:", X_test.shape)   # (40, 13) (260, 13)
 
def design_matrix(X):                      # from Lab 4
    return np.c_[np.ones(len(X)), X]
 
def fit_normal_equation(X, y):             # from Lab 4
    A = design_matrix(X)
    return np.linalg.solve(A.T @ A, A.T @ y)
 
def rmse(y_true, y_pred):                  # from Lab 4
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
 
labels = ['intercept'] + features
 
# ---------- "Why this lab exists": raw fit vs column sd ----------
theta_raw = fit_normal_equation(X_train, y_train)
print(f"{'feature':14s}{'coefficient':>13}{'column sd':>12}")
for name, coef, s in zip(features, theta_raw[1:], X_train.std(axis=0)):
    print(f"{name:14s}{coef:13.4f}{s:12.4f}")
 
 
# ================= Task 1: Standardisation =================
 
# ---------- Task 1(a) ----------
def standardise(X_train, X_test):
    # mu and sigma come from the TRAIN set only; the test set is shifted/scaled
    # with those same two vectors (using the test set's own stats would be leakage).
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    Z_train = (X_train - mu) / sigma
    Z_test = (X_test - mu) / sigma
    return Z_train, Z_test, mu, sigma
 
Z_train, Z_test, mu, sigma = standardise(X_train, X_test)
 
# ---------- Task 1(b) ----------
print("\nPer-column std after standardising")
print(f"{'feature':14s}{'train':>10}{'test':>10}")
for name, s_tr, s_te in zip(features, Z_train.std(axis=0), Z_test.std(axis=0)):
    print(f"{name:14s}{s_tr:10.3f}{s_te:10.3f}")
# train is exactly 1 everywhere; test is close to 1 but not equal, which is the sign
# that the test set really was scaled with the training mu/sigma.
 
 
# ================= Task 2: Ridge regression =================
 
# ---------- Task 2(a) ----------
def fit_ridge(X, y, lam):
    A = design_matrix(X)                   # ones column goes on AFTER standardising
    P = np.eye(A.shape[1])
    P[0, 0] = 0                            # intercept is not penalised
    return np.linalg.solve(A.T @ A + lam * P, A.T @ y)
 
# sanity check: lam = 0 must reproduce the plain normal-equation fit on Z
print("\nridge(lam=0) == normal equation on Z:",
      np.allclose(fit_ridge(Z_train, y_train, 0), fit_normal_equation(Z_train, y_train)))
 
# ---------- Task 2(b) ----------
show_lams = [0, 1, 10, 100]
show_thetas = [fit_ridge(Z_train, y_train, lam) for lam in show_lams]
 
print(f"\n{'term':<14}" + "".join(f"{'lam=' + str(l):>12}" for l in show_lams))
for j, name in enumerate(labels):
    print(f"{name:<14}" + "".join(f"{th[j]:>12.4f}" for th in show_thetas))
 
print("\nExact zeros among the 13 feature coefficients (|coef| < 1e-8):")
for lam, th in zip(show_lams, show_thetas):
    n_zero = int(np.sum(np.abs(th[1:]) < 1e-8))
    print(f"  lam = {lam:<4} -> {n_zero} zeros")
 
 
# ================= Task 3: Choosing lambda by cross-validation =================
 
# ---------- Task 3(a) ----------
def k_fold_cv(X, y, lam, k=5, seed=0):
    n = len(y)
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, k)        # k groups, no overlap, nothing dropped
 
    fold_rmse = []
    held_out = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
 
        theta = fit_ridge(X[train_idx], y[train_idx], lam)
        pred = design_matrix(X[val_idx]) @ theta
        fold_rmse.append(rmse(y[val_idx], pred))
        held_out.append(val_idx)
 
    # every row must be held out exactly once
    assert np.array_equal(np.sort(np.concatenate(held_out)), np.arange(n))
    return np.mean(fold_rmse)
 
# ---------- Task 3(b) ----------
lam_grid = [0, 0.1, 0.3, 1, 3, 10, 30, 100]
cv_scores = [k_fold_cv(Z_train, y_train, lam) for lam in lam_grid]
 
print(f"\n{'lambda':>8}{'CV RMSE':>12}")
for lam, s in zip(lam_grid, cv_scores):
    print(f"{lam:>8}{s:>12.4f}")
best_lam = lam_grid[int(np.argmin(cv_scores))]
print("best lambda by CV:", best_lam)
 
# ---------- Task 3(c): touch the test set once, only now ----------
theta_ols = fit_ridge(Z_train, y_train, 0)
theta_best = fit_ridge(Z_train, y_train, best_lam)
test_rmse_ols = rmse(y_test, design_matrix(Z_test) @ theta_ols)
test_rmse_best = rmse(y_test, design_matrix(Z_test) @ theta_best)
print(f"\nTest RMSE at lam = 0         : {test_rmse_ols:.4f}")
print(f"Test RMSE at lam = {best_lam:<9}: {test_rmse_best:.4f}")
 