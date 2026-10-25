# -*- coding: utf-8 -*-
"""Flight Delay Analysis


Runs in Google Colab, VS Code, or any Python environment.
Put the dataset (full_data_flightdelay.csv or .csv.zip) in a 'data/' folder
next to this script, or pass its path:  python flight_delay_analysis.py path/to/file

Colab notebook version:
    https://colab.research.google.com/drive/1vCxG5px5kuvCBiEBiSFHmVmS1cI77Y11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
import sys
import webbrowser

IN_COLAB = 'google.colab' in sys.modules

# Load dataset: pass a path as an argument, or put the file in ./data/
_candidates = [
    'data/full_data_flightdelay.csv.zip',
    'data/full_data_flightdelay.csv',
    'full_data_flightdelay.csv.zip',
    'full_data_flightdelay.csv',
    '/content/full_data_flightdelay.csv.zip',   # Google Colab upload location
    '/content/full_data_flightdelay.csv',
]
if len(sys.argv) > 1 and not sys.argv[1].startswith('-') and os.path.exists(sys.argv[1]):
    DATA_PATH = sys.argv[1]
else:
    DATA_PATH = next((p for p in _candidates if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "Dataset not found. Download it from "
        "https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations "
        "and place full_data_flightdelay.csv (or .csv.zip) in a 'data/' folder next to this script."
    )
df = pd.read_csv(DATA_PATH)

print("Rows in loaded file:", len(df))

# Sample down to a maximum of 500,000 rows if the file is larger, using stratified sampling
max_rows_for_comfort = 500000
if len(df) > max_rows_for_comfort:
    # Calculate the fraction to sample to get max_rows_for_comfort
    sample_fraction = max_rows_for_comfort / len(df)
    df, _ = train_test_split(df, train_size=sample_fraction, stratify=df['DEP_DEL15'], random_state=42)
    df = df.reset_index(drop=True)

# Impute missing values for 'AWND' if any
if df['AWND'].isnull().sum() > 0:
    awnd_median = df['AWND'].median()
    df['AWND'] = df['AWND'].fillna(awnd_median)
    print(f"Missing values in 'AWND' imputed with median: {awnd_median}")

print("="*60)
print("AIRLINE FLIGHT DELAY DATASET")
print("="*60)

print("\nFirst Five Records")
print(df.head())

print("\nShape of Dataset")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows Check")
dup_count = df.duplicated().sum()
print(f"Number of duplicate rows: {dup_count}")

if dup_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicates removed. New shape: {df.shape}")

print("\nSummary Statistics (Numerical)")
print(df.describe())

print("\nSummary Statistics (Categorical)")
print(df.describe(include='object'))

print("\nMedian")
print(df.median(numeric_only=True))

print("\nVariance")
print(df.var(numeric_only=True))

print("\nStandard Deviation")
print(df.std(numeric_only=True))

print("\nSkewness")
print(df.skew(numeric_only=True))

print("\nKurtosis")
print(df.kurt(numeric_only=True))

print("\nOutlier Detection (IQR method)")
outlier_cols = ['PLANE_AGE', 'AWND', 'TMAX']

for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers found (bounds: {lower:.2f} to {upper:.2f})")
    df[col] = df[col].clip(lower, upper)

print("Outliers capped using IQR method.")

df_raw = df.copy()
label_encoder = LabelEncoder()

df['CARRIER_NAME'] = label_encoder.fit_transform(df['CARRIER_NAME'])
df['DEP_TIME_BLK'] = label_encoder.fit_transform(df['DEP_TIME_BLK'])
df['DEPARTING_AIRPORT'] = label_encoder.fit_transform(df['DEPARTING_AIRPORT'])
df['PREVIOUS_AIRPORT'] = label_encoder.fit_transform(df['PREVIOUS_AIRPORT'])

print("\nAfter Label Encoding")
print(df[['CARRIER_NAME', 'DEP_TIME_BLK', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT']].head())

features = pd.get_dummies(
    df_raw.drop(columns=['DEP_DEL15', 'PREVIOUS_AIRPORT']),
    columns=['CARRIER_NAME', 'DEP_TIME_BLK', 'DEPARTING_AIRPORT'],
    drop_first=True, dtype=int
)
features['HAS_PREV_AIRPORT'] = (df_raw['PREVIOUS_AIRPORT'] != 'NONE').astype(int)
y = df['DEP_DEL15']

X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, stratify=y, random_state=42
)

standard_scaler = StandardScaler()
X_train = pd.DataFrame(standard_scaler.fit_transform(X_train),
                       columns=features.columns, index=X_train.index)
X_test = pd.DataFrame(standard_scaler.transform(X_test),
                      columns=features.columns, index=X_test.index)

print("\nStandardized Training Features")
print(X_train.head())

print("\nCorrelation Matrix")
print(df.corr())

plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), cmap='coolwarm', center=0)
plt.title("Correlation Heatmap")
plt.show()

print("Insight:")
print("No single feature shows a strong correlation with DEP_DEL15, suggesting delays depend on a combination of factors rather than one dominant cause.")
print("Note: correlations for label-encoded columns (carrier, time block, airports) are not meaningful, as their numeric codes have no real order.")

print("\nFeature Matrix Shape: ", X_train.shape)
print("\nTarget Vector Shape: ", y_train.shape)

df.to_csv('flightdelay_preprocessed.csv', index=False)
print("\nPreprocessed Dataset Saved Successfully")

print("\n" + "="*60)
print("MATPLOTLIB VISUALIZATIONS")
print("="*60)

# Line Plot
plt.figure(figsize=(8,5))
monthly_delay = df.groupby('MONTH')['DEP_DEL15'].mean()
plt.plot(monthly_delay.index, monthly_delay.values, marker='o')
plt.title("Delay Rate by Month")
plt.xlabel("Month")
plt.ylabel("Delay Rate")
plt.grid(True)
plt.show()
print("Insight: Delay rate varies noticeably across months, indicating a seasonal pattern.")

# Bar Plot
plt.figure(figsize=(8,5))
df_raw['CARRIER_NAME'].value_counts().plot(kind='bar')
plt.title("Flight Count by Carrier")
plt.xlabel("Carrier")
plt.ylabel("Count")
plt.show()
print("Insight: Shows the distribution of flights across different carriers.")

# Histogram
plt.figure(figsize=(8,5))
plt.hist(df['TMAX'].dropna(), bins=20)
plt.title("Max Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.show()
print("Insight: Shows the distribution of max temperature on flight days.")

# Scatter Plot
plt.figure(figsize=(8,5))
plt.scatter(df['PLANE_AGE'], df['CONCURRENT_FLIGHTS'])
plt.title("Plane Age vs Concurrent Flights")
plt.xlabel("Plane Age")
plt.ylabel("Concurrent Flights")
plt.show()
print("Insight: Visualizes relationship between aircraft age and airport congestion.")

# Pie Chart
plt.figure(figsize=(7,7))
df['DEP_DEL15'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Flight Delay Distribution")
plt.ylabel("")
plt.show()
print("Insight: Displays percentage of delayed versus on-time flights.")

# Box Plot
plt.figure(figsize=(6,5))
plt.boxplot(df['TMAX'].dropna())
plt.title("Box Plot of Max Temperature")
plt.show()
print("Insight: Shows the spread and outliers in temperature values.")

print("\n" + "="*60)
print("SEABORN VISUALIZATIONS")
print("="*60)

sns.set(style="whitegrid")

# Line Plot
plt.figure(figsize=(8,5))
sns.lineplot(x=monthly_delay.index, y=monthly_delay.values)
plt.title("Delay Rate by Month")
plt.show()
print("Insight: Seaborn gives a smoother view of the monthly delay trend.")

# Bar Plot
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='DEP_DEL15')
plt.title("Flight Delay Count")
plt.show()
print("Insight: Confirms the class imbalance between delayed and on-time flights.")

# Histogram
plt.figure(figsize=(8,5))
sns.histplot(df['TMAX'], bins=20, kde=True)
plt.title("Max Temperature Distribution")
plt.show()
print("Insight: KDE curve shows the spread of temperature values.")

# Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='PLANE_AGE', y='CONCURRENT_FLIGHTS')
plt.title("Plane Age vs Concurrent Flights")
plt.show()
print("Insight: Shows how aircraft age relates to congestion at departure.")

# Box Plot
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='DEP_DEL15', y='TMAX')
plt.title("Max Temperature vs Delay")
plt.show()
print("Insight: Compares temperature spread between delayed and on-time flights.")

# Pair Plot
pair_cols = ['DEP_DEL15', 'TMAX', 'PRCP', 'AWND', 'PLANE_AGE']
pairplot_sample = df[pair_cols].sample(n=5000, random_state=42)
sns.pairplot(pairplot_sample, hue='DEP_DEL15')
plt.show()
print("Insight: Shows how weather and plane age relate to each other and to delay status (visualized on a 5,000-row sample for performance; patterns hold across the full dataset since sampling is random).")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

print("\n" + "="*60)
print("CLASSIFICATION MODELS")
print("="*60)

def evaluate_model(model_name, y_true, y_pred):
    print("\n" + "="*60)
    print(model_name)
    print("="*60)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Accuracy  :", round(accuracy, 4))
    print("Precision :", round(precision, 4))
    print("Recall    :", round(recall, 4))
    print("F1-Score  :", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["On-Time", "Delayed"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    return accuracy, precision, recall, f1

# ---- Logistic Regression ----
logreg_model = LogisticRegression(max_iter=500, class_weight='balanced')
logreg_model.fit(X_train, y_train)
logreg_pred = logreg_model.predict(X_test)
logreg_results = evaluate_model("Logistic Regression", y_test, logreg_pred)
print("Insight: Logistic Regression models a linear relationship between features and delay probability, giving a simple baseline.")

# ---- Decision Tree ----
dtree_model = DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42)
dtree_model.fit(X_train, y_train)
dtree_pred = dtree_model.predict(X_test)
dtree_results = evaluate_model("Decision Tree", y_test, dtree_pred)
print("Insight: Decision Tree can capture non-linear interactions between features, such as weather combined with airport congestion.")

# ---- KNN ----
# Sample down just for KNN training (KNN scales poorly with large training sets)
knn_sample_size = 50000
X_train_knn = X_train.sample(n=knn_sample_size, random_state=42)
y_train_knn = y_train.loc[X_train_knn.index]

knn_model = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
knn_model.fit(X_train_knn, y_train_knn)
knn_pred = knn_model.predict(X_test)
knn_results = evaluate_model("K-Nearest Neighbors (KNN)", y_test, knn_pred)
print("Insight: KNN performance depends heavily on feature scaling (already applied here) and the choice of k. Trained on a 50,000-row sample of the training set for computational feasibility, as KNN scales poorly with dataset size.")

# ---- Model Comparison ----
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'KNN'],
    'Accuracy': [logreg_results[0], dtree_results[0], knn_results[0]],
    'Precision': [logreg_results[1], dtree_results[1], knn_results[1]],
    'Recall': [logreg_results[2], dtree_results[2], knn_results[2]],
    'F1 Score': [logreg_results[3], dtree_results[3], knn_results[3]]
})

print(comparison_df.round(4))

comparison_df.set_index('Model')[['Accuracy','Precision','Recall','F1 Score']].plot(
    kind='bar', figsize=(10, 5)
)
plt.title("Model Comparison - Logistic Regression vs Decision Tree vs KNN")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
print("Insight: Since about 19% of flights are delayed, Recall and F1 on the delayed class matter more than Accuracy alone when judging these models.")

# ---- Confusion Matrices (side-by-side) ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

models_predictions = [
    ("Logistic Regression", logreg_pred),
    ("Decision Tree", dtree_pred),
    ("KNN", knn_pred)
]

for ax, (name, predictions) in zip(axes, models_predictions):
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["On-Time", "Delayed"],
        yticklabels=["On-Time", "Delayed"]
    )
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

# ---- Best Model by F1-Score ----
best_model = comparison_df.loc[comparison_df['F1 Score'].idxmax()]
print("\nBest Model Based on F1-Score:")
print(best_model)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Note: This is a supporting analysis. AVG_MONTHLY_PASS_AIRPORT is largely fixed per airport, so it describes airport size rather than delays.")

print("\n" + "="*60)
print("REGRESSION: Predicting AVG_MONTHLY_PASS_AIRPORT")
print("="*60)

reg_target = 'AVG_MONTHLY_PASS_AIRPORT'
X_reg = df.drop(columns=[reg_target, 'DEP_DEL15'])
y_reg = df[reg_target]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_reg = StandardScaler()
X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
X_reg_test_scaled = scaler_reg.transform(X_reg_test)

# ---- Linear Regression ----
lin_reg = LinearRegression()
lin_reg.fit(X_reg_train_scaled, y_reg_train)
lin_pred = lin_reg.predict(X_reg_test_scaled)

lin_mae = mean_absolute_error(y_reg_test, lin_pred)
lin_mse = mean_squared_error(y_reg_test, lin_pred)
lin_rmse = np.sqrt(lin_mse)
lin_r2 = r2_score(y_reg_test, lin_pred)

print("\nLinear Regression Results")
print("MAE: ", lin_mae)
print("MSE: ", lin_mse)
print("RMSE: ", lin_rmse)
print("R2 Score: ", lin_r2)

# ---- Polynomial Regression (top 5 correlated features, degree 2) ----
correlations = df.drop(columns=['DEP_DEL15']).corr()[reg_target].abs().sort_values(ascending=False)
top_features = correlations.drop(reg_target).head(5).index.tolist()
print("\nTop 5 features selected for Polynomial Regression:", top_features)

X_poly_base = df[top_features]
X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(
    X_poly_base, y_reg, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train_expanded = poly.fit_transform(X_poly_train)
X_poly_test_expanded = poly.transform(X_poly_test)

poly_scaler = StandardScaler()
X_poly_train_scaled = poly_scaler.fit_transform(X_poly_train_expanded)
X_poly_test_scaled = poly_scaler.transform(X_poly_test_expanded)

poly_reg = LinearRegression()
poly_reg.fit(X_poly_train_scaled, y_poly_train)
poly_pred = poly_reg.predict(X_poly_test_scaled)

poly_mae = mean_absolute_error(y_poly_test, poly_pred)
poly_mse = mean_squared_error(y_poly_test, poly_pred)
poly_rmse = np.sqrt(poly_mse)
poly_r2 = r2_score(y_poly_test, poly_pred)

print("\nPolynomial Regression Results (degree=2)")
print("MAE: ", poly_mae)
print("MSE: ", poly_mse)
print("RMSE: ", poly_rmse)
print("R2 Score: ", poly_r2)

# ---- Comparison ----
print("\n" + "="*60)
print("REGRESSION MODEL COMPARISON")
print("="*60)

reg_comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Polynomial Regression (deg=2)'],
    'MAE': [lin_mae, poly_mae],
    'MSE': [lin_mse, poly_mse],
    'RMSE': [lin_rmse, poly_rmse],
    'R2 Score': [lin_r2, poly_r2]
})
print(reg_comparison)

plt.figure(figsize=(8,5))
reg_comparison.set_index('Model')[['MAE','RMSE','R2 Score']].plot(kind='bar')
plt.title("Linear vs Polynomial Regression — AVG_MONTHLY_PASS_AIRPORT")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ========== FINAL PROJECT INSIGHTS ==========

print("\n" + "="*60)
print("FINAL PROJECT INSIGHTS")
print("="*60)

# Dataset Insights
print("\n1. DATASET INSIGHTS")
delay_rate = df['DEP_DEL15'].mean() * 100
print(f"Total Flights Analyzed: {len(df)}")
print(f"Delayed Flights: {delay_rate:.2f}%")
print(f"On-Time Flights: {100-delay_rate:.2f}%")

# Seasonal Insights
print("\n2. FLIGHT DELAY PATTERNS")
print(f"Highest Delay Month: {monthly_delay.idxmax()}")
print(f"Lowest Delay Month: {monthly_delay.idxmin()}")
print("Flight delays vary across months, indicating seasonal patterns.")

# Classification Insights
print("\n3. CLASSIFICATION INSIGHTS")
best_clf = comparison_df.loc[comparison_df['F1 Score'].idxmax()]
print(f"Best Model: {best_clf['Model']}")
print(f"Accuracy: {best_clf['Accuracy']:.4f}")
print(f"Precision: {best_clf['Precision']:.4f}")
print(f"Recall: {best_clf['Recall']:.4f}")
print(f"F1 Score: {best_clf['F1 Score']:.4f}")
print("F1-score balances precision and recall for delay prediction.")

# Regression Insights
print("\n4. REGRESSION INSIGHTS")
best_reg = reg_comparison.loc[reg_comparison['RMSE'].idxmin()]
print(f"Best Model: {best_reg['Model']}")
print(f"RMSE: {best_reg['RMSE']:.4f}")
print(f"R2 Score: {best_reg['R2 Score']:.4f}")
print("Lower RMSE indicates better prediction accuracy.")

# Final Conclusion
print("\n5. FINAL CONCLUSION")
print("The project analyzed airline flight delays using")
print("data preprocessing, visualization, and machine learning.")
print(f"{best_clf['Model']} achieved the highest F1-score")
print("among the evaluated classification models.")
print(f"{best_reg['Model']} achieved the lowest RMSE")
print("for predicting average monthly airport passengers.")
print("The results demonstrate how machine learning can")
print("identify flight delay patterns and support prediction.")

print("\n" + "="*60)

# ============================================================
# FLIGHT DELAY WEBSITE - builds flight_delay_website.html.
# In Colab it shows an "Open in new tab" button; elsewhere it
# opens the page in your default browser.
# ============================================================
import json, base64
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def _show(html):
    """Colab: show the button widget. Local: open the saved .html file in the browser."""
    if IN_COLAB:
        from IPython.display import HTML, display
        display(HTML(html))
    else:
        path = os.path.abspath('flight_delay_website.html')
        print("Website saved to:", path)
        webbrowser.open('file://' + path)

need = ['df', 'df_raw', 'monthly_delay', 'comparison_df', 'reg_comparison',
        'logreg_pred', 'dtree_pred', 'knn_pred', 'y_test', 'top_features']
miss = [v for v in need if v not in globals()]
if miss:
    raise NameError(f"Run the full notebook first. Missing: {miss}")

MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
mn = lambda m: MON[int(m) - 1] if 1 <= int(m) <= 12 else str(m)

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    return o

# ---- statistics from your notebook ----
total = len(df_raw); rate = float(df['DEP_DEL15'].mean() * 100)
bc = comparison_df.loc[comparison_df['F1 Score'].idxmax()]
br = reg_comparison.loc[reg_comparison['RMSE'].idxmin()]

car = (df_raw.groupby('CARRIER_NAME')['DEP_DEL15'].agg(['count', 'mean'])
       .sort_values('count', ascending=False).head(12))
tb = df_raw.groupby('DEP_TIME_BLK')['DEP_DEL15'].agg(['count', 'mean']).sort_index()
ap = df_raw.groupby('DEPARTING_AIRPORT')['DEP_DEL15'].agg(['count', 'mean'])
ap = ap[ap['count'] >= max(200, 0.001 * total)].sort_values('mean', ascending=False).head(10)
cnt, ed = np.histogram(df_raw['TMAX'].dropna(), bins=20)
sc = df_raw[['PLANE_AGE', 'CONCURRENT_FLIGHTS']].dropna().sample(n=min(800, total), random_state=42)
g = df_raw.groupby('DEP_DEL15')['TMAX'].describe()
num = df_raw.select_dtypes(include='number')
corr = num.corr()['DEP_DEL15'].drop('DEP_DEL15').dropna().sort_values()
hm, lm = monthly_delay.idxmax(), monthly_delay.idxmin()
bigc = car[car['count'] >= 500] if (car['count'] >= 500).any() else car
wc = bigc['mean'].idxmax(); wb = tb['mean'].idxmax(); bb = tb['mean'].idxmin()

D = clean({
 'kpi': {'total': total, 'delayed': int(df['DEP_DEL15'].sum()), 'rate': rate,
         'wm': mn(hm), 'wmr': monthly_delay[hm] * 100, 'wb': wb, 'wbr': tb.loc[wb, 'mean'] * 100,
         'bc': bc['Model'], 'bcf': bc['F1 Score'], 'br': br['Model'], 'brr': br['RMSE']},
 'monthly': {'l': [mn(m) for m in monthly_delay.index], 'v': (monthly_delay.values * 100).round(2).tolist()},
 'car': {'l': car.index.tolist(), 'c': car['count'].tolist(), 'r': (car['mean'] * 100).round(2).tolist()},
 'tb': {'l': tb.index.tolist(), 'r': (tb['mean'] * 100).round(2).tolist()},
 'ap': {'l': ap.index.tolist(), 'r': (ap['mean'] * 100).round(2).tolist()},
 'hist': {'l': [f"{ed[i]:.0f}-{ed[i+1]:.0f}" for i in range(20)], 'c': cnt.tolist()},
 'sc': [{'x': float(a), 'y': float(b)} for a, b in zip(sc['PLANE_AGE'], sc['CONCURRENT_FLIGHTS'])],
 'tmax': [{'k': 'Delayed' if int(k) == 1 else 'On-time', 'n': r['count'], 'mean': r['mean'], 'std': r['std'],
           'min': r['min'], 'q1': r['25%'], 'med': r['50%'], 'q3': r['75%'], 'max': r['max']} for k, r in g.iterrows()],
 'stats': [{'c': c, 'mean': num[c].mean(), 'med': num[c].median(), 'std': num[c].std(), 'var': num[c].var(),
            'skew': num[c].skew(), 'kurt': num[c].kurt(), 'min': num[c].min(), 'max': num[c].max()} for c in num.columns],
 'corr': {'l': corr.index.tolist(), 'v': corr.round(4).tolist()},
 'clf': comparison_df.to_dict('records'),
 'cms': [{'n': n, 'm': confusion_matrix(y_test, p).tolist()} for n, p in
         [('Logistic Regression', logreg_pred), ('Decision Tree', dtree_pred), ('KNN', knn_pred)]],
 'reg': reg_comparison.to_dict('records'), 'poly': list(top_features),
 'ins': [
  f"{total:,} flights analysed: {rate:.1f}% delayed by 15+ minutes. The classes are imbalanced, so Recall and F1 matter more than Accuracy.",
  f"Delays are seasonal: highest in {mn(hm)} ({monthly_delay[hm]*100:.1f}%), lowest in {mn(lm)} ({monthly_delay[lm]*100:.1f}%).",
  f"Among carriers with 500+ flights, {wc} has the highest delay rate ({bigc.loc[wc,'mean']*100:.1f}%).",
  f"Worst departure block is {wb} ({tb.loc[wb,'mean']*100:.1f}%); best is {bb} ({tb.loc[bb,'mean']*100:.1f}%).",
  f"Strongest single correlation with delay is {corr.abs().idxmax()} (r = {corr[corr.abs().idxmax()]:.3f}), so delays come from a mix of factors.",
  f"Best classifier by F1: {bc['Model']} (F1 {bc['F1 Score']:.3f}, Recall {bc['Recall']:.3f}).",
  f"Best regressor by RMSE: {br['Model']} (R2 {br['R2 Score']:.3f}). A supporting analysis: passenger volume mostly describes airport size.",
 ]})

# ---- the website ----
T = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Flight Delay Analytics</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Anton&family=Space+Grotesk:wght@400;500;700&family=Caveat:wght@700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
@property --p{syntax:'<number>';inherits:false;initial-value:0}
:root{--A:Anton,Impact,'Arial Narrow',sans-serif;--bg:#ffffff;--tx:#0b2540;--mu:#5b7490;--bd:rgba(14,165,233,.24);--lime:#0ea5e9;--deep:#0284c7;--sky:#0ea5e9;--vio:#7dd3fc;--grn:#0284c7}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:70px}
body{margin:0;color:var(--tx);background:var(--bg);font:15px/1.6 'Space Grotesk',system-ui,sans-serif;overflow-x:hidden}
.bg{position:fixed;inset:0;z-index:-1;background:radial-gradient(60% 45% at 85% 0,rgba(56,189,248,.24),transparent),radial-gradient(40% 30% at 0 60%,rgba(125,211,252,.2),transparent),repeating-linear-gradient(0deg,rgba(14,165,233,.05) 0 1px,transparent 1px 64px),var(--bg)}
.intro{position:fixed;inset:0;z-index:100;background:var(--lime);display:grid;place-items:center;pointer-events:none;animation:up .9s 1.5s cubic-bezier(.7,0,.2,1) forwards}
.intro span{font-size:90px;color:#fff;animation:go 1.5s ease-in forwards}
@keyframes go{from{transform:translateX(-55vw) rotate(-8deg)}to{transform:translateX(55vw) rotate(-8deg)}}@keyframes up{to{transform:translateY(-101%)}}
.cur{position:fixed;left:0;top:0;width:16px;height:16px;border-radius:50%;background:rgba(14,165,233,.55);pointer-events:none;z-index:99;display:none;transition:width .2s,height .2s}
.trk{position:fixed;right:14px;top:80px;bottom:30px;width:2px;z-index:15;background:repeating-linear-gradient(var(--bd) 0 6px,transparent 6px 12px)}
.trk b{position:absolute;left:-9px;top:0;font-size:20px;color:var(--lime);transform:rotate(90deg)}
nav{position:sticky;top:0;z-index:20;display:flex;justify-content:space-between;align-items:center;gap:16px;padding:12px 28px;background:rgba(255,255,255,.84);backdrop-filter:blur(12px);border-bottom:1px solid var(--bd)}
.logo{font:400 24px var(--A);text-transform:uppercase;letter-spacing:1px;white-space:nowrap}.logo i{color:var(--lime);font-style:normal}
nav div{display:flex;gap:4px;overflow-x:auto}nav a{color:var(--mu);text-decoration:none;padding:7px 15px;border-radius:99px;font-size:13px;font-weight:700;white-space:nowrap}
nav a:hover{color:var(--deep)}nav a.on{background:var(--deep);color:#fff}
.hero{position:relative;max-width:1400px;min-height:calc(100vh - 56px);margin:0 auto;padding:28px 28px;display:flex;flex-direction:column;justify-content:space-between;overflow:hidden}
.hero h1{margin:5vh 0 0;font:400 clamp(90px,21vw,330px)/.82 var(--A);text-transform:uppercase;letter-spacing:-2px;position:relative;z-index:0}
.hero h1 span{display:block}.o{color:transparent;-webkit-text-stroke:2px var(--lime)}
.pw{position:absolute;z-index:1;right:-3vw;top:15vh;width:min(62vw,780px);pointer-events:none;transform:translate(calc(var(--sy,0)*1.2px),calc(var(--sy,0)*-.4px))}
.pw::before{content:'';position:absolute;right:62%;top:50%;width:80vw;height:7px;border-radius:9px;background:linear-gradient(90deg,transparent,rgba(14,165,233,.55));filter:blur(2px)}
.plane{display:block;width:100%;animation:bob 4s ease-in-out infinite alternate;filter:drop-shadow(0 30px 40px rgba(2,132,199,.35))}
@keyframes bob{from{transform:rotate(-6deg) translateY(0)}to{transform:rotate(-3deg) translateY(-16px)}}
.hb{position:relative;z-index:2;display:flex;justify-content:space-between;align-items:flex-end;gap:30px;flex-wrap:wrap}
.hb p{max-width:470px;color:var(--mu);font-size:17px;margin:10px 0 0}
.sig{font:700 36px Caveat,cursive;color:var(--lime);transform:rotate(-4deg);display:inline-block}
.btn{display:inline-block;margin:14px 10px 0 0;padding:14px 26px;border-radius:99px;font-weight:700;text-decoration:none;color:#fff;background:var(--deep);transition:.25s}
.btn:hover{transform:translateY(-3px) rotate(-2deg);box-shadow:0 10px 30px rgba(14,165,233,.4)}.btn.o{background:transparent;color:var(--tx);border:1px solid var(--bd);-webkit-text-stroke:0}
.gauge{text-align:center}.gauge p{margin:12px auto 0;font-size:13px}
.ring{--s:170px;--t:14px;position:relative;width:var(--s);height:var(--s);border-radius:50%;margin:0 auto;display:grid;place-items:center;background:conic-gradient(var(--sky),var(--vio) calc(var(--p)*1%),rgba(14,165,233,.12) 0);transition:--p 1.6s cubic-bezier(.2,.8,.2,1)}
.ring::before{content:'';position:absolute;inset:var(--t);border-radius:50%;background:#fff}.ring b{position:relative;font:400 calc(var(--s)/3.6) var(--A)}
.mq{background:var(--lime);color:#fff;overflow:hidden;white-space:nowrap;transform:rotate(-1.2deg);margin:10px -20px 0}
.mq div{display:inline-block;padding:14px 0;font:400 28px var(--A);text-transform:uppercase;animation:mq 40s linear infinite}@keyframes mq{to{transform:translateX(-50%)}}
main{max-width:1240px;margin:0 auto;padding:0 28px 60px}section{padding-top:100px}
.eb{color:var(--deep);font-weight:700;letter-spacing:3px;text-transform:uppercase;font-size:12px}
h2{margin:6px 0 10px;font:400 clamp(42px,7.5vw,104px)/.92 var(--A);text-transform:uppercase;overflow-wrap:anywhere}
h2 em{font-style:normal;color:transparent;-webkit-text-stroke:2px var(--lime)}
.sub{color:var(--mu);margin:0 0 32px;font-size:16px;max-width:620px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px}
.kpi,.card{background:#fff;border:1px solid var(--bd);border-radius:26px;box-shadow:0 12px 34px rgba(14,165,233,.09)}.kpi{padding:20px;transition:.3s}
.kpi:hover{background:var(--lime);color:#fff}.kpi:hover .l,.kpi:hover .s{color:#e0f2fe}
.ic{font-size:22px;margin-bottom:8px}.kpi .l{color:var(--mu);font-size:11.5px;font-weight:700;text-transform:uppercase;letter-spacing:1.2px}
.kpi .v{font:400 clamp(30px,3.4vw,46px)/1.05 var(--A);text-transform:uppercase;overflow-wrap:anywhere;margin:4px 0}.kpi .s{color:var(--mu);font-size:12.5px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}.card{padding:24px}
.card h3{margin:0 0 4px;font:400 22px var(--A);text-transform:uppercase;letter-spacing:.5px}.card .n{color:var(--mu);font-size:12.5px;margin:0 0 14px}
.w{grid-column:1/-1}.ch{position:relative;height:300px}.g2{grid-template-columns:repeat(2,1fr)}.g3{grid-template-columns:repeat(3,1fr)}.s2{grid-column:span 2}
.rv{opacity:0;transform:translateY(40px) scale(.98);transition:opacity .8s,transform .8s,border-color .3s}
.rv.in{opacity:1;transform:none;transition:opacity .8s,transform .3s,border-color .3s,background .3s,color .3s}.rv.in:hover{transform:translateY(-5px);border-color:var(--lime)}
.sc{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 14px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--bd)}
th:first-child,td:first-child{text-align:left}th{color:var(--mu);font-size:11.5px;text-transform:uppercase;letter-spacing:.8px}
tr.best td{background:rgba(14,165,233,.1)}tr.best td:first-child{color:var(--deep);font-weight:700}
.bar{position:relative;min-width:110px;height:22px;border-radius:6px;background:rgba(14,165,233,.1);text-align:right}.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:6px;background:linear-gradient(90deg,#38bdf8,#0284c7);opacity:.6}.bar span{position:relative;padding:0 8px;line-height:22px;font-weight:700}
.rings{display:flex;flex-wrap:wrap;gap:16px;justify-content:space-around;margin-top:20px;color:var(--mu);font-size:12px;text-align:center}.rings .ring{--s:104px;--t:10px;margin-bottom:6px}
.cm{display:grid;grid-template-columns:70px 1fr 1fr;gap:6px;text-align:center;font-size:12px}.cm .h{color:var(--mu);align-self:center}.cc{color:#0b2540;border-radius:14px;padding:16px 4px;font:400 20px var(--A)}.cc small{display:block;font:500 11px 'Space Grotesk',sans-serif;opacity:.85}
.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{padding:6px 16px;border-radius:99px;border:1px solid var(--bd);font-size:13px}
.ins{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}.ins .card{display:flex;gap:14px;padding:20px}
.num{flex:none;width:36px;height:36px;border-radius:50%;display:grid;place-items:center;font:400 18px var(--A);background:var(--lime);color:#fff}
footer{overflow:hidden;padding:70px 0 24px;text-align:center;color:var(--mu);font-size:13px}.bigf{font:400 clamp(60px,17vw,260px)/.9 var(--A);text-transform:uppercase;color:transparent;-webkit-text-stroke:2px var(--lime);white-space:nowrap}
@media(max-width:1000px){.g2,.g3{grid-template-columns:1fr}.s2{grid-column:auto}}
@media(max-width:860px){.grid{grid-template-columns:1fr}.pw{width:92vw;top:24vh}.trk{display:none}}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
</style></head><body><div class="bg"></div><div class="intro"><span>&#9992;</span></div><div class="cur" id="cur"></div><div class="trk"><b id="tp">&#9992;</b></div>
<nav><div class="logo">&#9992; Delay<i>.</i></div><div><a href="#overview">Overview</a><a href="#patterns">Patterns</a><a href="#stats">Statistics</a><a href="#clf">Classification</a><a href="#reg">Regression</a><a href="#insights">Insights</a></div></nav>
<header class="hero"><h1><span>Flight</span><span class="o">Delays</span></h1>
<div class="pw" id="pw"><svg class="plane" viewBox="0 0 330 120" aria-hidden="true"><path d="M28 44 8 6h46l30 36z" fill="#0ea5e9"/><path d="M22 62C22 44 62 40 100 40h150c44 2 70 12 78 22-8 10-34 20-78 22H100c-38 0-78-4-78-22z" fill="#fff" stroke="#0284c7" stroke-width="3"/><path d="M140 70l70 44h36l-40-44z" fill="#38bdf8"/><rect x="176" y="88" width="50" height="20" rx="10" fill="#94a3b8"/><path d="M290 52c14 1 26 4 34 8-8 3-20 6-34 7z" fill="#0b2540"/><path d="M70 62H260" stroke="#0b2540" stroke-width="9" stroke-linecap="round" stroke-dasharray="0.1 20"/></svg></div>
<div class="hb"><div><span class="sig">Flight Delay Analytics</span><p>Every statistic, pattern and model result from <b id="hn"></b> flights, ready for take-off.</p>
<a class="btn" href="#overview">Take off &#8595;</a><a class="btn o" href="#clf">Model results</a></div>
<div class="gauge"><div class="ring" id="hr"><b id="hv">0%</b></div><p>of flights delayed 15+ min</p></div></div></header>
<div class="mq"><div id="mq"></div></div>
<main>
<section id="overview"><div class="eb">01 / Overview</div><h2>The big picture</h2><p class="sub">Headline numbers from the cleaned dataset and the best models.</p><div class="kpis" id="kpis"></div></section>
<section id="patterns"><div class="eb">02 / Patterns</div><h2>When and where delays happen</h2><p class="sub">Seasonality, carriers, time of day, airports and weather.</p><div class="grid g3">
<div class="card rv s2" data-c="m"><h3>Delay rate by month</h3><p class="n">Larger dot marks the worst month</p><div class="ch"><canvas id="m"></canvas></div></div>
<div class="card rv" data-c="p"><h3>Delayed vs on-time</h3><p class="n">Class balance of the target</p><div class="ch"><canvas id="p"></canvas></div></div>
<div class="card rv w" data-c="c"><h3>Carriers: volume and delay rate</h3><p class="n">Top 12 carriers by number of flights</p><div class="ch"><canvas id="c"></canvas></div></div>
<div class="card rv" data-c="b"><h3>Delay rate by departure time block</h3><p class="n">Green is better, red is worse</p><div class="ch"><canvas id="b"></canvas></div></div>
<div class="card rv" data-c="a"><h3>Highest-delay departure airports</h3><p class="n">Only airports with enough flights are ranked</p><div class="ch"><canvas id="a"></canvas></div></div>
<div class="card rv" data-c="h"><h3>Max temperature distribution</h3><p class="n">20 equal-width bins</p><div class="ch"><canvas id="h"></canvas></div></div>
<div class="card rv" data-c="s"><h3>Plane age vs concurrent flights</h3><p class="n">Random sample of 800 flights</p><div class="ch"><canvas id="s"></canvas></div></div>
<div class="card rv s2" data-c="r"><h3>Correlation with delay</h3><p class="n">Label-encoded columns are excluded because their codes have no real order</p><div class="ch" style="height:520px"><canvas id="r"></canvas></div></div>
</div></section>
<section id="stats"><div class="eb">03 / Statistics</div><h2>Descriptive statistics</h2><p class="sub">Every numeric column, plus the box-plot numbers for temperature.</p><div class="grid">
<div class="card rv w"><h3>Summary statistics</h3><p class="n">Mean, median, spread and shape</p><div class="sc"><table id="t1"></table></div></div>
<div class="card rv w"><h3>Max temperature by delay status</h3><p class="n">Five-number summary</p><div class="sc"><table id="t2"></table></div></div></div></section>
<section id="clf"><div class="eb">04 / Classification</div><h2>Predicting delays</h2><p class="sub">Logistic Regression, Decision Tree and KNN compared. The winner is picked by F1-score.</p><div class="grid g2">
<div class="card rv" data-c="k"><h3>Model comparison</h3><p class="n">Accuracy, precision, recall and F1</p><div class="ch"><canvas id="k"></canvas></div></div>
<div class="card rv"><h3 id="bt">Best model</h3><p class="n">Scores on the held-out test set</p><div class="rings" id="rg"></div></div>
<div class="card rv w"><h3>All model scores</h3><p class="n">Best model highlighted</p><div class="sc"><table id="t3"></table></div></div>
<div class="card rv w"><h3>Confusion matrices</h3><p class="n">Rows are actual, columns are predicted</p><div class="grid g3" id="cms"></div></div></div></section>
<section id="reg"><div class="eb">05 / Regression</div><h2>Predicting AVG_MONTHLY_PASS_AIRPORT</h2><p class="sub">Linear vs polynomial (degree 2). The winner has the lowest RMSE.</p><div class="grid">
<div class="card rv w"><div class="sc"><table id="t4"></table></div></div>
<div class="card rv w"><h3>Polynomial model features</h3><p class="n">Top 5 by correlation with the target</p><div class="chips" id="pf"></div></div></div></section>
<section id="insights"><div class="eb">06 / Insights</div><h2>Key takeaways</h2><p class="sub">Written automatically from your results.</p><div class="ins" id="ins"></div></section>
</main><footer><div class="bigf">Bon voyage</div><p>Built from your Colab notebook</p></footer>
<script>
const D=__DATA__,$=i=>document.getElementById(i),K=D.kpi;
const nf=(n,d=0)=>n==null?'-':Number(n).toLocaleString(undefined,{maximumFractionDigits:d});
const f4=n=>n==null?'-':(Math.abs(n)>=1000?nf(n):Number(n).toFixed(4));
const sf=(n,d)=>n!=null&&n!==0&&Math.abs(n)<0.01?Number(n).toExponential(2):nf(n,d);
const C=['#0ea5e9','#0b2540','#7dd3fc','#fbbf24','#34d399','#fb7185'];
const heat=t=>{t=Math.max(0,Math.min(1,t));const g=[52,211,153],a=[251,191,36],r=[251,113,133],u=t<.5?t*2:(t-.5)*2,f=t<.5?g:a,o=t<.5?a:r;return 'rgb('+f.map((v,i)=>Math.round(v+(o[i]-v)*u))+')'};
const norm=a=>{const lo=Math.min(...a),hi=Math.max(...a);return a.map(v=>(v-lo)/((hi-lo)||1))};
const G=(id,a,b)=>{const g=$(id).getContext('2d').createLinearGradient(0,0,0,320);g.addColorStop(0,a);g.addColorStop(1,b);return g};
Chart.defaults.color='#5b7490';Chart.defaults.borderColor='rgba(14,165,233,.13)';Chart.defaults.font.family="'Space Grotesk',system-ui,sans-serif";
Chart.defaults.plugins.legend.labels.usePointStyle=true;
Object.assign(Chart.defaults.plugins.tooltip,{backgroundColor:'#0b2540',borderColor:'rgba(255,255,255,.15)',borderWidth:1,padding:12,cornerRadius:10,titleColor:'#fff'});
const mk=(id,c)=>{c.options=Object.assign({responsive:true,maintainAspectRatio:false},c.options||{});return new Chart($(id),c)};
const pc=v=>v+'%',noleg={legend:{display:false}},R=8;
const sh=function(v){const t=this.getLabelForValue(v);return t.length>20?t.slice(0,19)+'…':t};
const B={
m(){const v=D.monthly.v,mx=Math.max(...v);mk('m',{type:'line',data:{labels:D.monthly.l,datasets:[{data:v,borderColor:C[0],backgroundColor:G('m','rgba(14,165,233,.38)','rgba(14,165,233,0)'),fill:true,tension:.35,borderWidth:3,pointBackgroundColor:'#fff',pointRadius:v.map(x=>x===mx?8:4)}]},options:{plugins:noleg,scales:{y:{ticks:{callback:pc}}}}})},
p(){mk('p',{type:'doughnut',data:{labels:['On time','Delayed'],datasets:[{data:[K.total-K.delayed,K.delayed],backgroundColor:[C[4],C[5]],borderColor:'#fff',borderWidth:4,hoverOffset:10}]},options:{cutout:'66%',plugins:{legend:{position:'bottom'}}}})},
c(){mk('c',{data:{labels:D.car.l,datasets:[{type:'bar',label:'Flights',data:D.car.c,backgroundColor:G('c','rgba(14,165,233,.95)','rgba(125,211,252,.35)'),borderRadius:R,yAxisID:'y',order:2},{type:'line',label:'Delay rate %',data:D.car.r,borderColor:C[3],backgroundColor:C[3],yAxisID:'y1',tension:.3,borderWidth:3,order:1}]},options:{scales:{x:{ticks:{callback:sh}},y1:{position:'right',grid:{drawOnChartArea:false},ticks:{callback:pc}}}}})},
b(){mk('b',{type:'bar',data:{labels:D.tb.l,datasets:[{data:D.tb.r,backgroundColor:norm(D.tb.r).map(heat),borderRadius:R}]},options:{plugins:noleg,scales:{y:{ticks:{callback:pc}}}}})},
a(){mk('a',{type:'bar',data:{labels:D.ap.l,datasets:[{data:D.ap.r,backgroundColor:norm(D.ap.r).map(heat),borderRadius:R}]},options:{indexAxis:'y',plugins:noleg,scales:{x:{ticks:{callback:pc}},y:{ticks:{callback:sh}}}}})},
h(){mk('h',{type:'bar',data:{labels:D.hist.l,datasets:[{data:D.hist.c,backgroundColor:G('h','rgba(14,165,233,.95)','rgba(125,211,252,.3)'),borderRadius:6}]},options:{plugins:noleg,datasets:{bar:{categoryPercentage:1,barPercentage:.92}},scales:{x:{ticks:{maxRotation:60}}}}})},
s(){mk('s',{type:'scatter',data:{datasets:[{data:D.sc,backgroundColor:'rgba(14,165,233,.5)',pointRadius:4}]},options:{plugins:noleg,scales:{x:{title:{display:true,text:'Plane age'}},y:{title:{display:true,text:'Concurrent flights'}}}}})},
r(){mk('r',{type:'bar',data:{labels:D.corr.l,datasets:[{data:D.corr.v,backgroundColor:D.corr.v.map(v=>v>=0?C[0]:C[5]),borderRadius:6}]},options:{indexAxis:'y',plugins:noleg,scales:{y:{ticks:{autoSkip:false}}}}})},
k(){const M=['Accuracy','Precision','Recall','F1 Score'];mk('k',{type:'bar',data:{labels:M,datasets:D.clf.map((m,i)=>({label:m.Model,data:M.map(x=>m[x]),backgroundColor:C[i],borderRadius:R}))},options:{scales:{y:{min:0,max:1}},plugins:{legend:{position:'bottom'}}}})}};

// count-up numbers + KPI cards
const up=(el,to,dec,suf)=>{const t0=performance.now(),st=t=>{const p=Math.min(1,(t-t0)/1400),e=1-Math.pow(1-p,3);el.textContent=nf(to*e,dec)+suf;if(p<1)requestAnimationFrame(st)};requestAnimationFrame(st)};
const kp=[['🛫','Flights analysed',{n:K.total},'after cleaning','#0ea5e9','#6366f1'],['⏱','Delayed 15+ min',{n:K.rate,d:1,s:'%'},nf(K.delayed)+' flights','#f59e0b','#ef4444'],
['✅','On time',{n:100-K.rate,d:1,s:'%'},nf(K.total-K.delayed)+' flights','#10b981','#06b6d4'],['📅','Worst month',K.wm,nf(K.wmr,1)+'% delayed','#8b5cf6','#ec4899'],
['🕒','Worst time block',K.wb,nf(K.wbr,1)+'% delayed','#f43f5e','#f59e0b'],['🏆','Best classifier',K.bc,'F1 '+K.bcf.toFixed(3),'#14b8a6','#3b82f6']];
$('kpis').innerHTML=kp.map((x,i)=>'<div class="kpi rv"><div class="ic" style="--a:'+x[4]+';--b:'+x[5]+'">'+x[0]+'</div><div class="l">'+x[1]+'</div><div class="v" id="kv'+i+'">'+(typeof x[2]=='object'?'0':x[2])+'</div><div class="s">'+x[3]+'</div></div>').join('');
$('hn').textContent=nf(K.total);
const kicks=()=>kp.forEach((x,i)=>{if(typeof x[2]=='object')up($('kv'+i),x[2].n,x[2].d||0,x[2].s||'')});
setTimeout(()=>{$('hr').style.setProperty('--p',K.rate);up($('hv'),K.rate,1,'%');kicks()},1700);

// tables
const T=(id,h,rows,best)=>$(id).innerHTML='<thead><tr>'+h.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map((r,i)=>'<tr'+(i===best?' class="best"':'')+'>'+r.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('')+'</tbody>';
T('t1',['Column','Mean','Median','Std','Variance','Skew','Kurtosis','Min','Max'],D.stats.map(r=>[r.c,sf(r.mean,3),sf(r.med,3),sf(r.std,3),sf(r.var,2),nf(r.skew,3),nf(r.kurt,3),sf(r.min,2),sf(r.max,2)]));
T('t2',['Class','Count','Mean','Std','Min','Q1','Median','Q3','Max'],D.tmax.map(r=>[r.k,nf(r.n),nf(r.mean,2),nf(r.std,2),nf(r.min,1),nf(r.q1,1),nf(r.med,1),nf(r.q3,1),nf(r.max,1)]));
const M=['Accuracy','Precision','Recall','F1 Score'],bi=D.clf.reduce((b,m,i)=>m['F1 Score']>D.clf[b]['F1 Score']?i:b,0);
const bar=v=>'<div class="bar"><i style="width:'+(v*100)+'%"></i><span>'+v.toFixed(4)+'</span></div>';
T('t3',['Model'].concat(M),D.clf.map(m=>[m.Model].concat(M.map(x=>bar(m[x])))),bi);
$('bt').textContent='Best model: '+D.clf[bi].Model;
$('rg').innerHTML=M.map(x=>'<div><div class="ring" data-p="'+(D.clf[bi][x]*100)+'"><b>'+(D.clf[bi][x]*100).toFixed(0)+'</b></div>'+x+'</div>').join('');
D.cms.forEach(m=>{const [[a,b],[c,d]]=m.m,mx=Math.max(a,b,c,d)||1,r0=(a+b)||1,r1=(c+d)||1,
 cell=(v,t,ok)=>'<div class="cc" style="background:rgba('+(ok?'14,165,233':'251,113,133')+','+(.14+.6*v/mx).toFixed(2)+')">'+nf(v)+'<small>'+(100*v/t).toFixed(1)+'% of actual</small></div>',e=document.createElement('div');
 e.innerHTML='<h3 style="margin:0 0 10px;font-size:15px">'+m.n+'</h3><div class="cm"><div></div><div class="h">Pred. on-time</div><div class="h">Pred. delayed</div><div class="h">Actual on-time</div>'+cell(a,r0,1)+cell(b,r0,0)+'<div class="h">Actual delayed</div>'+cell(c,r1,0)+cell(d,r1,1)+'</div>';$('cms').appendChild(e)});
const ri=D.reg.reduce((b,m,i)=>m.RMSE<D.reg[b].RMSE?i:b,0);
T('t4',['Model','MAE','MSE','RMSE','R2 Score'],D.reg.map(m=>[m.Model,f4(m.MAE),f4(m.MSE),f4(m.RMSE),Number(m['R2 Score']).toFixed(4)]),ri);
$('pf').innerHTML=D.poly.map(f=>'<span class="chip">'+f+'</span>').join('');
$('ins').innerHTML=D.ins.map((t,i)=>'<div class="card rv"><div class="num">'+(i+1)+'</div><div>'+t+'</div></div>').join('');

// reveal on scroll: charts are drawn when they scroll into view
const show=e=>{e.classList.add('in');const b=B[e.dataset.c];if(b&&!e.done){e.done=1;b()}e.querySelectorAll('.ring[data-p]').forEach(r=>setTimeout(()=>r.style.setProperty('--p',r.dataset.p),200))};
const els=[...document.querySelectorAll('.rv')];
if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>es.forEach(x=>{if(x.isIntersecting){show(x.target);io.unobserve(x.target)}}),{threshold:.12});els.forEach(e=>io.observe(e))}else els.forEach(show);
const secs=[...document.querySelectorAll('main section')],lk=[...document.querySelectorAll('nav a')];
addEventListener('scroll',()=>{let c=secs[0].id;secs.forEach(s=>{if(s.getBoundingClientRect().top<160)c=s.id});lk.forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+c))},{passive:true});

// ---- plane-themed extras ----
document.querySelectorAll('h2').forEach(h=>{const w=h.textContent.split(' '),l=w.pop();h.innerHTML=(w.join(' ')+' <em>'+l+'</em>').trim()});
const half=(['&#9992; '+nf(K.total)+' flights','&#9992; '+K.rate.toFixed(1)+'% delayed','&#9992; worst month '+K.wm,'&#9992; worst block '+K.wb,'&#9992; best model '+K.bc].join('   ')+'   ').repeat(2);
$('mq').innerHTML=half+half;
const pw=$('pw'),tp=$('tp');
addEventListener('scroll',()=>{const y=scrollY,h=document.documentElement.scrollHeight-innerHeight;pw.style.setProperty('--sy',Math.min(y,innerHeight));tp.style.top=(h>0?y/h*100:0)+'%'},{passive:true});
if(matchMedia('(pointer:fine)').matches){const c=$('cur');c.style.display='block';
addEventListener('mousemove',e=>{c.style.transform='translate('+e.clientX+'px,'+e.clientY+'px) translate(-50%,-50%)'});
document.querySelectorAll('a,.card,.kpi').forEach(e=>{e.onmouseenter=()=>{c.style.width=c.style.height='46px'};e.onmouseleave=()=>{c.style.width=c.style.height='16px'}})}
</script></body></html>"""

page = T.replace('__DATA__', json.dumps(D).replace('</', '<\\/'))

# ---- open it in its own browser tab (no server needed) ----
with open('flight_delay_website.html', 'w', encoding='utf-8') as f:
    f.write(page)

b64 = base64.b64encode(page.encode('utf-8')).decode()
_show("""
<div style="font-family:system-ui,sans-serif;padding:20px;border-radius:14px;background:#0b1224;color:#eaf0ff;display:flex;gap:12px;flex-wrap:wrap;align-items:center">
  <b style="font-size:16px;margin-right:8px">&#9992; Your website is ready</b>
  <button id="op" style="padding:11px 20px;border:0;border-radius:10px;font-weight:700;color:#fff;cursor:pointer;background:linear-gradient(90deg,#0ea5e9,#8b5cf6)">Open in new tab</button>
  <button id="dl" style="padding:11px 20px;border:1px solid #334;border-radius:10px;font-weight:700;color:#fff;cursor:pointer;background:#16203a">Download .html</button>
</div>
<script>
(function(){
  const b64="__B64__";
  const blob=()=>new Blob([Uint8Array.from(atob(b64),c=>c.charCodeAt(0))],{type:'text/html'});
  document.getElementById('op').onclick=()=>window.open(URL.createObjectURL(blob()),'_blank');
  document.getElementById('dl').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(blob());a.download='flight_delay_website.html';a.click()};
})();
</script>""".replace('__B64__', b64))
print("Also saved as flight_delay_website.html in the current folder.")