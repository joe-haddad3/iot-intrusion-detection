import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)

df = pd.read_csv('data/Merged01.csv')

print(f"Shape: {df.shape}")

# ── 0. Basic info (ADDED) ────────────────────────────────────────────────────
print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- SAMPLE ROWS ---")
print(df.head())

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())


# ── 1. Missing values ─────────────────────────────────────────────────────────
missing = df.isnull().sum()
print("\n--- MISSING VALUES ---")
if missing.sum() == 0:
    print("No missing values found.")
else:
    print(missing[missing > 0])


# ── 2. Infinite values ────────────────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns
inf_counts = np.isinf(df[numeric_cols]).sum()
print("\n--- INFINITE VALUES ---")
if inf_counts.sum() == 0:
    print("No infinite values found.")
else:
    print(inf_counts[inf_counts > 0])


# ── 3. Duplicate rows ─────────────────────────────────────────────────────────
n_dupes = df.duplicated().sum()
print(f"\n--- DUPLICATES ---")
print(f"Duplicate rows: {n_dupes} ({n_dupes / len(df) * 100:.2f}%)")


# ── 4. Constant columns ───────────────────────────────────────────────────────
constant_cols = [col for col in numeric_cols if df[col].nunique() == 1]
print(f"\n--- CONSTANT COLUMNS ---")
if constant_cols:
    print(f"Columns with only one unique value: {constant_cols}")
else:
    print("No constant columns found.")


# ── 5. Outliers (IQR method) ──────────────────────────────────────────────────
print("\n--- OUTLIERS (IQR method) ---")
outlier_summary = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
    if n_outliers > 0:
        outlier_summary[col] = n_outliers

print(f"Features with outliers: {len(outlier_summary)} / {len(numeric_cols)}")
for col, count in sorted(outlier_summary.items(), key=lambda x: -x[1])[:10]:
    print(f"  {col}: {count} outliers ({count/len(df)*100:.1f}%)")


# ── 6. Class balance ──────────────────────────────────────────────────────────
print("\n--- CLASS BALANCE ---")
counts = df['Label'].value_counts()
print(counts.to_string())

# (ADDED) percentages
print("\n--- CLASS DISTRIBUTION (%) ---")
print((counts / len(df) * 100).round(2))

plt.figure(figsize=(14, 6))
counts.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Attack Type')
plt.ylabel('Samples')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/class_distribution.png')
plt.show()


# ── 7. Correlation analysis (ADDED) ───────────────────────────────────────────
print("\n--- CORRELATION ANALYSIS ---")
corr = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, aspect='auto')
plt.colorbar()
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig('outputs/correlation_matrix.png')


# ── 8. Feature distributions (ADDED) ──────────────────────────────────────────
print("\n--- FEATURE DISTRIBUTIONS ---")
df_finite = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
df_finite.hist(figsize=(15, 10))
plt.tight_layout()
plt.savefig('outputs/feature_histograms.png')


print("\nAll exploration outputs saved in 'outputs/' folder.")