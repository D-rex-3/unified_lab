# ================= IC-272 Lab 2: Pandas and a First Pipeline =================
import numpy as np
import pandas as pd

df = pd.read_csv("lab2_machines.csv")   # (300, 6)

# ---------- carried over from Lab 1 (must already be correct) ----------
def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[TN, FP], [FN, TP]])

def precision_recall_f1(cm):
    TN, FP = cm[0]; FN, TP = cm[1]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
    return precision, recall, f1

def accuracy_score_scratch(y_true, y_pred):
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))

# ---------- Block 1, Task 1: Inspect ----------
print(df.isna().sum())                        # missing values per column
print(df['needs_service'].value_counts())     # class distribution
print(df['machine_type'].value_counts())      # category distribution
print(df.groupby('needs_service').apply(lambda g: g.isna().sum()))  # -> which class loses more values?

# ---------- Block 1, Task 2: Global vs group-wise imputation ----------
global_median = df['temperature'].median()
group_medians = df.groupby('needs_service')['temperature'].median()
df['temperature_global'] = df['temperature'].fillna(global_median)
df['temperature_groupwise'] = df['temperature'].fillna(
    df.groupby('needs_service')['temperature'].transform('median')
)
print("global median:", global_median)
print("group-wise medians:\n", group_medians)

# ---------- Block 1, Task 3: One-hot encoding by hand ----------
categories = sorted(df['machine_type'].unique())
manual_ohe = pd.DataFrame({
    f"machine_type_{c}": (df['machine_type'] == c).astype(int) for c in categories
})
auto_ohe = pd.get_dummies(df['machine_type']).astype(int)
auto_ohe.columns = [f"machine_type_{c}" for c in auto_ohe.columns]
auto_ohe = auto_ohe[manual_ohe.columns]
print("manual one-hot matches pd.get_dummies:", manual_ohe.equals(auto_ohe))

# ---------- Block 1, Task 4: train_test_split ----------
def train_test_split(X, y, test_size=0.2, seed=42):
    X = np.asarray(X); y = np.asarray(y)
    n = X.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    n_test = int(n * test_size)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_dummy = df[['runtime_hours']].to_numpy()
y_dummy = df['needs_service'].to_numpy()
Xtr1, Xte1, ytr1, yte1 = train_test_split(X_dummy, y_dummy, 0.2, 42)
Xtr2, Xte2, ytr2, yte2 = train_test_split(X_dummy, y_dummy, 0.2, 42)
print("same seed -> identical split:", np.array_equal(Xtr1, Xtr2))

# ---------- Block 1, Task 5 (stretch): stratified split ----------
def stratified_train_test_split(X, y, test_size=0.2, seed=42):
    X = np.asarray(X); y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for c in np.unique(y):
        idx_c = rng.permutation(np.where(y == c)[0])
        n_test_c = int(len(idx_c) * test_size)
        test_parts.append(idx_c[:n_test_c])
        train_parts.append(idx_c[n_test_c:])
    train_idx = rng.permutation(np.concatenate(train_parts))
    test_idx = rng.permutation(np.concatenate(test_parts))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

Xtr_s, Xte_s, ytr_s, yte_s = stratified_train_test_split(X_dummy, y_dummy, 0.2, 42)
print("full / train / test positive rate:", y_dummy.mean(), ytr_s.mean(), yte_s.mean())

# ---------- Block 2: The leakage-free pipeline ----------
raw = pd.read_csv("lab2_machines.csv")
num_cols = ['runtime_hours', 'temperature', 'vibration', 'pressure']
cat_col = 'machine_type'
y_col = 'needs_service'

n = len(raw)
perm = np.random.default_rng(42).permutation(n)
n_test = int(n * 0.2)
test_idx, train_idx = perm[:n_test], perm[n_test:]

train_df = raw.iloc[train_idx].reset_index(drop=True).copy()
test_df = raw.iloc[test_idx].reset_index(drop=True).copy()
y_train = train_df.pop(y_col).to_numpy()
y_test = test_df.pop(y_col).to_numpy()

# 3: medians from TRAIN ONLY, applied to both splits
train_medians = train_df[num_cols].median()
train_df[num_cols] = train_df[num_cols].fillna(train_medians)
test_df[num_cols] = test_df[num_cols].fillna(train_medians)

# 4: categories seen in TRAIN ONLY
train_categories = sorted(train_df[cat_col].unique())
for c in train_categories:
    train_df[f"{cat_col}_{c}"] = (train_df[cat_col] == c).astype(int)
    test_df[f"{cat_col}_{c}"] = (test_df[cat_col] == c).astype(int)
    # a category that appears in test but never in train just gets all-zero columns here
train_df = train_df.drop(columns=[cat_col])
test_df = test_df.drop(columns=[cat_col])

# 5: majority-class baseline from TRAIN ONLY
majority_class = pd.Series(y_train).mode()[0]
y_pred_baseline = np.full_like(y_test, fill_value=majority_class)

# 6: report with Lab 1 functions
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
precision, recall, f1 = precision_recall_f1(cm_baseline)
accuracy = accuracy_score_scratch(y_test, y_pred_baseline)
print(f"BASELINE accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
print(cm_baseline)
