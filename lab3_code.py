# ================= IC-272 Lab 3: Visualization =================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- carried over from Lab 1 ----------
def column_stats(X):
    return X.mean(axis=0), X.std(axis=0), X.min(axis=0), X.max(axis=0)

def pairwise_distances(A, B):
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))

def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[TN, FP], [FN, TP]])

def multiclass_confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    K = classes.size
    idx = {c: i for i, c in enumerate(classes)}
    ti = np.array([idx[c] for c in y_true]); pi = np.array([idx[c] for c in y_pred])
    cm = np.zeros((K, K), dtype=int)
    np.add.at(cm, (ti, pi), 1)
    return cm, classes

def precision_recall_f1(cm):
    TN, FP = cm[0]; FN, TP = cm[1]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
    return precision, recall, f1

# ---------- Block 0: Setup ----------
feat_df = pd.read_csv("lab1_features.csv")
pred_df = pd.read_csv("lab1_predictions.csv")   # ADJUST column names below if yours differ
mach_df = pd.read_csv("lab2_machines.csv")

X = feat_df.to_numpy()
cols = feat_df.columns

# ---------- Block 1, Task 1: Five distributions ----------
mean, std, minimum, maximum = column_stats(X)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for i, ax in enumerate(axes.flat):
    if i >= X.shape[1]:
        ax.axis('off'); continue
    ax.hist(X[:, i], bins=20, color='steelblue', edgecolor='white')
    ax.axvline(mean[i], color='red', label='mean')
    ax.axvline(mean[i] - std[i], color='red', linestyle='--', label='mean ± 1 std')
    ax.axvline(mean[i] + std[i], color='red', linestyle='--')
    ax.set_title(cols[i]); ax.set_xlabel(cols[i]); ax.set_ylabel('count')
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("lab3_distributions.png", dpi=150)
plt.show()
# the column with the largest raw spread (std) will dominate a Euclidean distance calc

# ---------- Block 1, Task 2: Distance matrix as an image ----------
A_, B_ = X[:50], X[50:70]
D = pairwise_distances(A_, B_)
nn = D.argmin(axis=1)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(D, aspect='auto', cmap='viridis')
fig.colorbar(im, ax=ax, label='distance')
ax.scatter(nn, np.arange(D.shape[0]), color='red', marker='x', s=30, label='nearest neighbour')
ax.set_xlabel('row index in B'); ax.set_ylabel('row index in A')
ax.set_title('Pairwise distances, A vs B'); ax.legend()
fig.savefig("lab3_distance_matrix.png", dpi=150)
plt.show()

# ---------- Block 1, Task 3: Confusion matrix as an image ----------
y_true4 = pred_df['y_true_multi'].to_numpy()
y_pred4 = pred_df['y_pred_multi'].to_numpy()
cm4, classes4 = multiclass_confusion_matrix(y_true4, y_pred4)

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(cm4, cmap='Blues')
fig.colorbar(im, ax=ax)
for i in range(cm4.shape[0]):
    for j in range(cm4.shape[1]):
        ax.text(j, i, cm4[i, j], ha='center', va='center',
                color='white' if cm4[i, j] > cm4.max() / 2 else 'black')
ax.set_xticks(range(len(classes4))); ax.set_xticklabels(classes4)
ax.set_yticks(range(len(classes4))); ax.set_yticklabels(classes4)
ax.set_xlabel('predicted label'); ax.set_ylabel('true label')
ax.set_title('Multiclass confusion matrix')
fig.savefig("lab3_confusion_matrix.png", dpi=150)
plt.show()
# an all-zero column = that class was never predicted, no matter what the true label was

# ---------- Block 2, Task 1: Where the holes are ----------
missing_rate = mach_df.groupby('needs_service')[['temperature','vibration','pressure']] \
                       .apply(lambda g: g.isna().mean())
features = ['temperature', 'vibration', 'pressure']
classes = missing_rate.index.to_numpy()
x = np.arange(len(features)); width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
for i, c in enumerate(classes):
    ax.bar(x + i * width, missing_rate.loc[c, features], width, label=f'class {c}')
ax.set_xticks(x + width / 2); ax.set_xticklabels(features)
ax.set_ylabel('missing-value rate'); ax.set_title('Missingness by class'); ax.legend()
fig.savefig("lab3_missingness.png", dpi=150)
plt.show()

# ---------- Block 2, Task 2: What imputation does to a distribution ----------
temp = mach_df['temperature']
global_median = temp.median()
group_medians = mach_df.groupby('needs_service')['temperature'].median()
temp_dropped = temp.dropna()
temp_global = temp.fillna(global_median)
temp_group = temp.fillna(mach_df.groupby('needs_service')['temperature'].transform('median'))

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(temp_dropped, bins=30, alpha=0.5, label='dropped NA')
ax.hist(temp_global, bins=30, alpha=0.5, label='global-median imputed')
ax.hist(temp_group, bins=30, alpha=0.5, label='group-wise imputed')
ax.axvline(global_median, color='black', linestyle='--', label='global median')
for c, m in group_medians.items():
    ax.axvline(m, linestyle=':', label=f'class {c} median')
ax.set_xlabel('temperature'); ax.set_ylabel('count'); ax.legend()
ax.set_title('Effect of imputation on the temperature distribution')
fig.savefig("lab3_imputation_effect.png", dpi=150)
plt.show()
# the global-median bars grow a spike at the global median that isn't in the real data

# ---------- Block 2, Task 3: Do the classes separate? ----------
fig, ax = plt.subplots(figsize=(7, 6))
for c, marker, color in zip([0, 1], ['o', '^'], ['tab:blue', 'tab:orange']):
    sub = mach_df[mach_df['needs_service'] == c]
    ax.scatter(sub['temperature'], sub['vibration'], marker=marker, color=color,
               alpha=0.6, label=f'needs_service={c}')
ax.set_xlabel('temperature'); ax.set_ylabel('vibration')
ax.set_title('vibration vs temperature by class'); ax.legend()
fig.savefig("lab3_scatter.png", dpi=150)
plt.show()
# eyeball a threshold t_guess here, e.g. 75, to test in Block 3

# ---------- Block 3, Task 1: Threshold sweep (global imputation) ----------
def f1_sweep(feature_values, y_true, thresholds):
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred = (feature_values > t).astype(int)
        p, r, f = precision_recall_f1(confusion_matrix(y_true, y_pred))
        precisions.append(p); recalls.append(r); f1s.append(f)
    return np.array(precisions), np.array(recalls), np.array(f1s)

y_true_ns = mach_df['needs_service'].to_numpy()
thresholds = np.linspace(temp_global.min(), temp_global.max(), 200)
p_g, r_g, f1_g = f1_sweep(temp_global.to_numpy(), y_true_ns, thresholds)
best_t_g = thresholds[np.argmax(f1_g)]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, p_g, label='precision')
ax.plot(thresholds, r_g, label='recall')
ax.plot(thresholds, f1_g, label='F1')
ax.axvline(best_t_g, color='black', linestyle='--', label=f'best t = {best_t_g:.2f}')
ax.set_xlabel('threshold t'); ax.set_ylabel('score')
ax.set_title('Precision / Recall / F1 vs threshold (global imputation)'); ax.legend()
fig.savefig("lab3_sweep_global.png", dpi=150)
plt.show()

# ---------- Block 3, Task 2: Same sweep, group-wise imputation ----------
p_gw, r_gw, f1_gw = f1_sweep(temp_group.to_numpy(), y_true_ns, thresholds)
best_t_gw = thresholds[np.argmax(f1_gw)]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, f1_g, label='F1 (global imputation)')
ax.plot(thresholds, f1_gw, label='F1 (group-wise imputation)')
ax.set_xlabel('threshold t'); ax.set_ylabel('F1')
ax.set_title('Global vs group-wise imputation: F1 curves'); ax.legend()
fig.savefig("lab3_sweep_compare.png", dpi=150)
plt.show()
print("best F1 global:", f1_g.max(), "at t =", best_t_g)
print("best F1 groupwise:", f1_gw.max(), "at t =", best_t_gw)

# ---------- Block 3, Task 3 (stretch): vibration instead of temperature ----------
vib = mach_df['vibration']
vib_global = vib.fillna(vib.median())
vib_group = vib.fillna(mach_df.groupby('needs_service')['vibration'].transform('median'))
thresholds_v = np.linspace(vib_global.min(), vib_global.max(), 200)
_, _, f1_v_g = f1_sweep(vib_global.to_numpy(), y_true_ns, thresholds_v)
_, _, f1_v_gw = f1_sweep(vib_group.to_numpy(), y_true_ns, thresholds_v)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds_v, f1_v_g, label='vibration, global imputation')
ax.plot(thresholds_v, f1_v_gw, label='vibration, group-wise imputation')
ax.set_xlabel('threshold t'); ax.set_ylabel('F1')
ax.set_title('vibration as the one-rule feature'); ax.legend()
fig.savefig("lab3_sweep_vibration.png", dpi=150)
plt.show()
