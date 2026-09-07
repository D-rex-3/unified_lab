# ================= IC-272 Lab 4: Linear Regression from Scratch =================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Task 1(a): Load & inspect ----------
df = pd.read_csv("ic272_lab4_rent.csv")   # (300, 6)
print(df.describe().loc[['min', 'max']])

feature_cols = ['area', 'rooms', 'age', 'dist_km', 'floors']
X_all = df[feature_cols].to_numpy()
y_all = df['rent'].to_numpy()

# ---------- Task 1(b): train_test_split_scratch ----------
def train_test_split_scratch(X, y, test_frac, seed):
    n = X.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split_scratch(X_all, y_all, test_frac=0.2, seed=0)

# ---------- Task 1(c): simple linear regression, closed form ----------
def fit_simple_lr(x, y):
    x_bar, y_bar = x.mean(), y.mean()
    w = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
    b = y_bar - w * x_bar
    return w, b

area_idx = feature_cols.index('area')
x_train_area, x_test_area = X_train[:, area_idx], X_test[:, area_idx]
w_simple, b_simple = fit_simple_lr(x_train_area, y_train)
print(f"w = {w_simple:.3f}, b = {b_simple:.3f}")

# ---------- Task 1(d): plot ----------
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(x_test_area, y_test, alpha=0.6, label='test data')
xs = np.linspace(x_test_area.min(), x_test_area.max(), 100)
ax.plot(xs, w_simple * xs + b_simple, color='red', label='fitted line')
ax.set_xlabel('area'); ax.set_ylabel('rent')
ax.set_title('Simple linear regression: rent vs area'); ax.legend()
fig.savefig("lab4_simple_lr.png", dpi=150)
plt.show()

# ---------- Task 2(a): design matrix ----------
def design_matrix(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

# ---------- Task 2(b): normal equation ----------
def fit_normal_equation(X, y):
    A = design_matrix(X)
    # np.linalg.solve(A^T A, A^T y) solves the linear system directly instead of
    # forming (A^T A)^-1 explicitly -- it's numerically more stable and cheaper,
    # since computing a full matrix inverse is more work (and more error-prone)
    # than solving one system of equations for theta.
    return np.linalg.solve(A.T @ A, A.T @ y)

theta_hat = fit_normal_equation(X_train, y_train)

# ---------- Task 2(c): print as a table ----------
labels = ['intercept'] + feature_cols
print(f"{'term':<10}{'theta':>10}")
for name, val in zip(labels, theta_hat):
    print(f"{name:<10}{val:>10.3f}")

# ---------- Task 3(a): batch gradient descent ----------
def fit_gradient_descent(X, y, lr, n_iters):
    A = design_matrix(X)
    n = A.shape[0]
    theta = np.zeros(A.shape[1])
    losses = []
    for _ in range(n_iters):
        err = A @ theta - y
        grad = (2 / n) * (A.T @ err)
        theta = theta - lr * grad
        losses.append((1 / n) * np.sum(err ** 2))
    return theta, losses

# ---------- Task 3(b): run and compare to normal equation ----------
theta_gd, losses_gd = fit_gradient_descent(X_train, y_train, lr=0.005, n_iters=20000)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(losses_gd)
ax.set_xlabel('iteration'); ax.set_ylabel('J(theta)')
ax.set_title('Gradient descent loss curve (lr=0.005)')
fig.savefig("lab4_loss_curve.png", dpi=150)
plt.show()

print(f"{'term':<10}{'theta_hat (NE)':>16}{'theta_gd':>14}")
for name, ne, gd in zip(labels, theta_hat, theta_gd):
    print(f"{name:<10}{ne:>16.3f}{gd:>14.3f}")
print("largest |difference|:", np.max(np.abs(theta_hat - theta_gd)))

# ---------- Task 3(c): learning-rate sweep ----------
fig, ax = plt.subplots(figsize=(8, 5))
for lr in [0.001, 0.005, 0.01]:
    _, losses = fit_gradient_descent(X_train, y_train, lr=lr, n_iters=20000)
    ax.plot(losses, label=f'lr={lr}')
ax.set_yscale('log')
ax.set_xlabel('iteration'); ax.set_ylabel('J(theta) [log scale]')
ax.set_title('Effect of learning rate on convergence'); ax.legend()
fig.savefig("lab4_lr_sweep.png", dpi=150)
plt.show()
# the largest lr will overflow and print RuntimeWarnings -- expected divergence, not a bug

# ---------- Task 4(a): RMSE and R^2 ----------
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score_scratch(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot

A_train, A_test = design_matrix(X_train), design_matrix(X_test)
y_pred_train, y_pred_test = A_train @ theta_gd, A_test @ theta_gd

print(f"Train RMSE: {rmse(y_train, y_pred_train):.4f}   Train R2: {r2_score_scratch(y_train, y_pred_train):.4f}")
print(f"Test  RMSE: {rmse(y_test, y_pred_test):.4f}   Test  R2: {r2_score_scratch(y_test, y_pred_test):.4f}")

# ---------- Task 4(b): residual plot ----------
residuals = y_test - y_pred_test
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_pred_test, residuals, alpha=0.6)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('predicted rent'); ax.set_ylabel('residual (actual - predicted)')
ax.set_title('Residuals vs predicted values')
fig.savefig("lab4_residuals.png", dpi=150)
plt.show()
