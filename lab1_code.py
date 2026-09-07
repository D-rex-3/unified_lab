# ================= IC-272 Lab 1: NumPy and Pandas Refresher =================
import numpy as np
import pandas as pd

# ---------- Block 0: Setup ----------
print("numpy:", np.__version__)
print("pandas:", pd.__version__)

df_feat = pd.read_csv("lab1_features.csv")   # expected shape used later in Lab 3: (120, 5)
X = df_feat.to_numpy()

# ---------- Block 1, Task 1: Column summary ----------
def column_stats(X):
    """Mean, std, min, max of each column of a 2D array. No loops."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    minimum = X.min(axis=0)
    maximum = X.max(axis=0)
    return mean, std, minimum, maximum

mean, std, minimum, maximum = column_stats(X)
print("mean:", mean); print("std:", std)
print("min:", minimum); print("max:", maximum)

# ---------- Block 1, Task 2: Masking ----------
def clip_above(X, t):
    """Copy of X with every element > t replaced by t. Leaves X unmodified."""
    X_clipped = X.copy()
    X_clipped[X_clipped > t] = t
    return X_clipped

def count_above_per_row(X, t):
    """Count of elements > t, per row."""
    return (X > t).sum(axis=1)

t = float(np.median(X))
X_clipped = clip_above(X, t)
row_counts = count_above_per_row(X, t)
print("X left unmodified:", np.max(X) != np.max(X_clipped) or np.all(X <= t))
print("row counts (first 10):", row_counts[:10])

# ---------- Block 1, Task 3: Pairwise distances ----------
def pairwise_distances(A, B):
    """Euclidean distance between every row of A and every row of B.
    A: (n, d), B: (m, d) -> D: (n, m). Broadcasting only, no loops."""
    diff = A[:, None, :] - B[None, :, :]     # (n, 1, d) - (1, m, d) -> (n, m, d)
    return np.sqrt((diff ** 2).sum(axis=2))  # (n, m)

A_ = X[:10]
B_ = X[10:16]
D = pairwise_distances(A_, B_)
print("D shape:", D.shape)

D_naive = np.zeros((A_.shape[0], B_.shape[0]))
for i in range(A_.shape[0]):
    for j in range(B_.shape[0]):
        D_naive[i, j] = np.sqrt(np.sum((A_[i] - B_[j]) ** 2))
print("matches naive double loop:", np.allclose(D, D_naive))

# ---------- Block 1, Task 4: Nearest neighbour ----------
def nearest_neighbor_index(A, B):
    """For each row of A, index of the closest row of B. One np.argmin call."""
    D = pairwise_distances(A, B)
    return D.argmin(axis=1)     # length n

print("nearest neighbour indices:", nearest_neighbor_index(A_, B_))

# ---------- Block 2, Task 1: Binary confusion matrix ----------
def confusion_matrix(y_true, y_pred):
    """2x2 matrix for labels in {0,1}. Class 1 = positive.
    Rows = true label, columns = predicted label: [[TN, FP], [FN, TP]]."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[TN, FP], [FN, TP]])

rng = np.random.default_rng(0)
y_true_demo = rng.integers(0, 2, 50)
y_pred_demo = rng.integers(0, 2, 50)
cm = confusion_matrix(y_true_demo, y_pred_demo)
print("confusion matrix:\n", cm)
print("counts sum to n:", cm.sum() == len(y_true_demo))

# ---------- Block 2, Task 2: Precision / Recall / F1 ----------
def precision_recall_f1(cm):
    TN, FP = cm[0]; FN, TP = cm[1]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1

precision, recall, f1 = precision_recall_f1(cm)
print(f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

# ---------- Block 2, Task 3: Multiclass ----------
def multiclass_confusion_matrix(y_true, y_pred):
    """K x K matrix, K inferred from the data (not hard-coded)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    K = classes.size
    idx = {c: i for i, c in enumerate(classes)}
    true_idx = np.array([idx[c] for c in y_true])
    pred_idx = np.array([idx[c] for c in y_pred])
    cm = np.zeros((K, K), dtype=int)
    np.add.at(cm, (true_idx, pred_idx), 1)
    return cm, classes

def per_class_precision_recall_f1(cm):
    K = cm.shape[0]
    precisions = np.zeros(K); recalls = np.zeros(K); f1s = np.zeros(K)
    for c in range(K):
        TP = cm[c, c]
        FP = cm[:, c].sum() - TP
        FN = cm[c, :].sum() - TP
        precisions[c] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recalls[c] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1s[c] = (2 * precisions[c] * recalls[c] / (precisions[c] + recalls[c])
                  if (precisions[c] + recalls[c]) > 0 else 0.0)
    return precisions, recalls, f1s, f1s.mean()

y_true_multi = rng.integers(0, 4, 200)
y_pred_multi = rng.integers(0, 4, 200)
cm_multi, classes = multiclass_confusion_matrix(y_true_multi, y_pred_multi)
precisions, recalls, f1s, macro_f1 = per_class_precision_recall_f1(cm_multi)
print("multiclass CM:\n", cm_multi)
print("macro F1:", macro_f1)

# ---------- Block 2, Task 4: Verification against sklearn ----------
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, classification_report

print("binary matches sklearn:", np.array_equal(cm, sk_confusion_matrix(y_true_demo, y_pred_demo)))
print("multiclass matches sklearn:", np.array_equal(cm_multi, sk_confusion_matrix(y_true_multi, y_pred_multi)))
print(classification_report(y_true_multi, y_pred_multi, digits=4))
