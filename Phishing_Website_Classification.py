# -*- coding: utf-8 -*-
# Phishing Website Classification — Group 11, Section 26920
# Atheer AlSulami, Dima Albaity, Danah Alamri, Mawaddah Asiri

# 1. Load Dataset

# Import libraries and load the dataset.
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve, roc_auc_score
)
from joblib import dump

RANDOM_STATE = 42
DATA_PATH = Path('PhishingData.arff')

if not DATA_PATH.exists():
    try:
        zip_path = Path('website_phishing_uci.zip')
        urlretrieve('https://archive.ics.uci.edu/static/public/379/website%2Bphishing.zip', zip_path)
        with ZipFile(zip_path) as zf:
            names = [name for name in zf.namelist() if name.lower().endswith('.arff')]
            if not names:
                raise FileNotFoundError('The UCI ZIP does not contain an ARFF file.')
            DATA_PATH.write_bytes(zf.read(names[0]))
        print('Dataset downloaded from UCI.')
    except Exception as error:
        print('UCI download unavailable:', type(error).__name__)
        print('Trying the matching GitHub repository copy...')
        urlretrieve(
            'https://raw.githubusercontent.com/adibamba/phishing-classification/main/PhishingData.arff',
            DATA_PATH
        )
        print('Dataset downloaded from GitHub.')

raw_data, metadata = arff.loadarff(str(DATA_PATH))
df = pd.DataFrame(raw_data)
for column in df.columns:
    df[column] = pd.to_numeric(
        df[column].map(lambda v: v.decode('utf-8') if isinstance(v, bytes) else v),
        errors='coerce'
    )
print('Loaded dataset:', DATA_PATH)
print('Shape (rows, columns):', df.shape)
assert df.shape[1] == 10, 'Unexpected column count; check that the correct dataset was loaded.'
assert 'Result' in df.columns, 'The target Result column is missing.'
assert set(df['Result'].dropna().unique()).issubset({-1, 0, 1}), 'Unexpected target labels.'

# Display the first rows and dataset information.
print('First 5 rows (head):')
display(df.head())
print('Last 5 rows (tail):')
display(df.tail())
print('Rows and columns:', df.shape)
print('\nData types:')
print(df.dtypes)
print('\nInformation:')
df.info()
print('\nDescription:')
display(df.describe().T)

# 2. Dataset Overview

# Check missing values, duplicates, and class counts.
print('Null / NaN per column:')
print(df.isnull().sum())
print('Total missing:', int(df.isnull().sum().sum()))
print('Missing-value markers (?, unknown) detected in loaded numeric data: 0 if all columns decoded properly.')
print('\nUnique values for every attribute:')
for column in df.columns:
    print(f'{column}: {sorted(df[column].dropna().unique().tolist())}')

print('\nExact duplicate rows:', int(df.duplicated().sum()))
print('\nClass distribution:')
class_labels = {-1: 'Phishing', 0: 'Suspicious', 1: 'Legitimate'}
print(df['Result'].value_counts().sort_index().rename(index=class_labels))

# 3. Data Visualization

# Plot feature histograms and class distribution.
feature_columns = [column for column in df.columns if column != 'Result']
df[feature_columns].hist(bins=3, figsize=(13, 10))
plt.suptitle('Histograms of Website Phishing Features')
plt.tight_layout()
plt.show()

df['Result'].map(class_labels).value_counts().reindex(
    ['Phishing', 'Suspicious', 'Legitimate']
).plot(kind='bar', figsize=(7, 4))
plt.title('Target Class Distribution')
plt.xlabel('Class')
plt.ylabel('Number of websites')
plt.xticks(rotation=0)
plt.show()

# Plot a boxplot and a feature scatter plot.
df[feature_columns].boxplot(figsize=(13, 5), rot=40)
plt.title('Boxplot of Encoded Website Features')
plt.ylabel('Feature code (-1, 0, 1)')
plt.show()

rng = np.random.default_rng(RANDOM_STATE)
fig, ax = plt.subplots(figsize=(8, 5))
for label, name in class_labels.items():
    part = df.loc[df['Result'] == label]
    ax.scatter(part['URL_Length'] + rng.normal(0, 0.05, len(part)),
               part['SSLfinal_State'] + rng.normal(0, 0.05, len(part)),
               label=name, alpha=0.40, s=16)
ax.set(xlabel='URL_Length (encoded)', ylabel='SSLfinal_State (encoded)',
       title='Scatter: URL Length vs. SSL Final State')
ax.legend()
plt.show()

# 4. Correlation Analysis

# Calculate and visualize feature correlations.
corr_matrix = df.corr(numeric_only=True)
display(corr_matrix.round(2))
plt.figure(figsize=(11, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Website Phishing Dataset')
plt.tight_layout()
plt.show()
print('Correlation with target Result:')
print(corr_matrix['Result'].drop('Result').sort_values(ascending=False))

# 5. Data Preprocessing

# Remove ambiguous records and duplicates.
X_columns = [column for column in df.columns if column != 'Result']

label_count_per_pattern = df.groupby(X_columns, dropna=False)['Result'].transform('nunique')
ambiguous_rows = label_count_per_pattern > 1
print('Rows with ambiguous feature/target patterns:', int(ambiguous_rows.sum()))
print('Exact duplicated rows before cleaning:', int(df.duplicated().sum()))

df_model = df.loc[~ambiguous_rows].drop_duplicates().reset_index(drop=True)
print('Original dataset shape:', df.shape)
print('Modeling dataset after ambiguity/duplicate cleaning:', df_model.shape)
print('Remaining class distribution:')
print(df_model['Result'].value_counts().sort_index())
assert df_model['Result'].nunique() == 3, 'One or more target classes disappeared after cleaning.'
assert not df_model[X_columns].duplicated().any(), 'Feature patterns still duplicate after cleaning.'

# Split the data into training and testing sets.
X = df_model.drop('Result', axis=1)
y = df_model['Result'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print('Training shape:', X_train.shape)
print('Testing shape:', X_test.shape)
print('Training class counts:')
print(y_train.value_counts().sort_index())
print('Test class counts:')
print(y_test.value_counts().sort_index())
assert set(map(tuple, X_train.values)).isdisjoint(set(map(tuple, X_test.values))),     'Found identical feature vectors in train and test!'

# 6. Feature Selection

# Select the top five features.
exploration_imputer = SimpleImputer(strategy='most_frequent')
X_train_imputed = exploration_imputer.fit_transform(X_train)
feature_selector = RFE(
    estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
    n_features_to_select=5
)
feature_selector.fit(X_train_imputed, y_train)
selected_features = X_train.columns[feature_selector.support_].tolist()
print('Selected 5 features (RFE trained on training data only):')
print(selected_features)
print('\nFeature rankings (1 means selected):')
display(pd.DataFrame({
    'Feature': X_train.columns,
    'Selected': feature_selector.support_,
    'RFE Ranking': feature_selector.ranking_
}).sort_values(['RFE Ranking', 'Feature']))

# 7. Model Training

# Train models and run cross-validation.
def make_pipeline(classifier, needs_scaling=False):
    steps = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('feature_selection', RFE(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            n_features_to_select=5
        ))
    ]
    if needs_scaling:
        steps.append(('scaler', StandardScaler()))
    steps.append(('classifier', classifier))
    return Pipeline(steps)

models = {
    'Decision Tree': make_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE)),
    'Random Forest': make_pipeline(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    'KNN': make_pipeline(KNeighborsClassifier(n_neighbors=5), needs_scaling=True)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name, model in models.items():
    macro_f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
    acc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_rows.append({
        'Model': name,
        'CV F1 Macro Mean': macro_f1_scores.mean(),
        'CV F1 Macro Std': macro_f1_scores.std(),
        'CV Accuracy Mean': acc_scores.mean()
    })
cv_table = pd.DataFrame(cv_rows).sort_values('CV F1 Macro Mean', ascending=False)
print('Cross-validation comparison (TRAINING data only):')
display(cv_table.round(3))
best_model_name = cv_table.iloc[0]['Model']
print('Selected by CV macro F1:', best_model_name)

# 8. Model Evaluation

# Evaluate each model and plot confusion matrices.
test_rows = []
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    test_rows.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision Macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall Macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'F1 Macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'Phishing Recall': recall_score(y_test == -1, y_pred == -1, zero_division=0)
    })

test_table = pd.DataFrame(test_rows).sort_values('F1 Macro', ascending=False)
print('Held-out test set results:')
display(test_table.round(3))

for name, y_pred in predictions.items():
    print('\n' + '=' * 55)
    print(name, '— Classification Report')
    print(classification_report(
        y_test, y_pred,
        labels=[-1, 0, 1],
        target_names=['Phishing', 'Suspicious', 'Legitimate'],
        zero_division=0
    ))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, y_pred, labels=[-1, 0, 1]),
        display_labels=['Phishing', 'Suspicious', 'Legitimate']
    )
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix — {name}')
    plt.tight_layout()
    plt.show()

# Compare the three models.
ax = test_table.set_index('Model')[['Accuracy', 'F1 Macro', 'Phishing Recall']].plot(
    kind='bar', figsize=(10, 5)
)
ax.set_ylabel('Score (0–1)')
ax.set_ylim(0, 1)
ax.set_title('Test-set Model Comparison')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print('Best model SELECTED using training CV macro F1:', best_model_name)
print('Its held-out test metrics:')
display(test_table.loc[test_table['Model'] == best_model_name].round(3))

# 9. Precision–Recall and ROC Curves

# Plot precision–recall and ROC curves.
best_model = models[best_model_name]
positive_class_index = list(best_model.named_steps['classifier'].classes_).index(-1)
y_binary = (y_test.to_numpy() == -1)
phishing_scores = best_model.predict_proba(X_test)[:, positive_class_index]

precisions, recalls, _ = precision_recall_curve(y_binary, phishing_scores)
fpr, tpr, _ = roc_curve(y_binary, phishing_scores)
auc_value = roc_auc_score(y_binary, phishing_scores)

plt.figure(figsize=(7, 5))
plt.plot(recalls, precisions)
plt.title(f'Precision–Recall: Phishing vs Other — {best_model_name}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.grid(alpha=0.2)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc_value:.3f}')
plt.plot([0, 1], [0, 1], '--')
plt.title(f'ROC: Phishing vs Other — {best_model_name}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(alpha=0.2)
plt.show()

# 10. Save Results

# Save the results and predict one example.
cv_table.to_csv('phishing_cv_results.csv', index=False)
test_table.to_csv('phishing_test_results.csv', index=False)
dump(best_model, 'phishing_best_model.joblib')

example_features = X_test.iloc[[0]]
example_prediction = int(best_model.predict(example_features)[0])
print('Input features:')
display(example_features)
print('Predicted class:', example_prediction, class_labels[example_prediction])
print('Actual class:', int(y_test.iloc[0]), class_labels[int(y_test.iloc[0])])
print('Saved: phishing_cv_results.csv, phishing_test_results.csv, phishing_best_model.joblib')

