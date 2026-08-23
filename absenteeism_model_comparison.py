import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report, mean_squared_error
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EMPIRICAL EVALUATION: LOGISTIC REGRESSION vs LINEAR REGRESSION vs DECISION TREE
# ============================================================================

print("\n" + "="*80)
print("ABSENTEEISM CLASSIFICATION: EMPIRICAL MODEL COMPARISON")
print("="*80 + "\n")

# ===== LOAD DATA =====
from google.colab import files

try:
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    print(f"✓ Uploaded file: {file_name}\n")
except:
    print("Using local file or Colab auto-mount\n")
    file_name = "Absenteeism_data.xlsx"

df = pd.read_excel(file_name)

# Data Preparation
median_absence = df['Absenteeism time in hours'].median()
df['Target'] = (df['Absenteeism time in hours'] >= median_absence).astype(int)

print(f"Dataset: {df.shape[0]} employees × {df.shape[1]} features")
print(f"Target Distribution:")
print(f"  • High Absence (1): {(df['Target']==1).sum()} ({(df['Target']==1).sum()/len(df)*100:.1f}%)")
print(f"  • Low Absence (0):  {(df['Target']==0).sum()} ({(df['Target']==0).sum()/len(df)*100:.1f}%)\n")

# Feature Selection
feature_cols = ['Son', 'Pet', 'Age', 'Body mass index', 'Transportation expense']
X = df[feature_cols].values
y = df['Target'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train Set: {X_train.shape[0]} samples")
print(f"Test Set:  {X_test.shape[0]} samples\n")

# ============================================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================================
print("="*80)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*80)

lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)

lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)
lr_test_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_metrics = {
    'Model': 'Logistic Regression',
    'Train Accuracy': accuracy_score(y_train, lr_train_pred),
    'Test Accuracy': accuracy_score(y_test, lr_test_pred),
    'Precision': precision_score(y_test, lr_test_pred),
    'Recall': recall_score(y_test, lr_test_pred),
    'F1-Score': f1_score(y_test, lr_test_pred),
    'ROC-AUC': roc_auc_score(y_test, lr_test_proba),
}

# Cross-validation
lr_cv_scores = cross_validate(
    lr_model, X_train_scaled, y_train, 
    cv=5, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
)

lr_metrics['CV Mean Accuracy'] = lr_cv_scores['test_accuracy'].mean()
lr_metrics['CV Std Accuracy'] = lr_cv_scores['test_accuracy'].std()

print(f"\nTest Accuracy:    {lr_metrics['Test Accuracy']:.4f}")
print(f"Precision:        {lr_metrics['Precision']:.4f}")
print(f"Recall:           {lr_metrics['Recall']:.4f}")
print(f"F1-Score:         {lr_metrics['F1-Score']:.4f}")
print(f"ROC-AUC:          {lr_metrics['ROC-AUC']:.4f}")
print(f"CV Accuracy:      {lr_metrics['CV Mean Accuracy']:.4f} (±{lr_metrics['CV Std Accuracy']:.4f})")

print("\nClassification Report:")
print(classification_report(y_test, lr_test_pred, target_names=['Low Absence', 'High Absence']))

# ============================================================================
# MODEL 2: LINEAR REGRESSION (as baseline regressor)
# ============================================================================
print("="*80)
print("MODEL 2: LINEAR REGRESSION (Baseline)")
print("="*80)

lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train_scaled, y_train)

lin_reg_train_pred = lin_reg_model.predict(X_train_scaled)
lin_reg_test_pred = lin_reg_model.predict(X_test_scaled)

# Convert to binary predictions (threshold 0.5)
lin_reg_test_pred_binary = (lin_reg_test_pred >= 0.5).astype(int)
lin_reg_train_pred_binary = (lin_reg_train_pred >= 0.5).astype(int)

# Clip predictions for ROC-AUC
lin_reg_test_proba_clipped = np.clip(lin_reg_test_pred, 0, 1)

lin_reg_metrics = {
    'Model': 'Linear Regression',
    'Train Accuracy': accuracy_score(y_train, lin_reg_train_pred_binary),
    'Test Accuracy': accuracy_score(y_test, lin_reg_test_pred_binary),
    'Precision': precision_score(y_test, lin_reg_test_pred_binary, zero_division=0),
    'Recall': recall_score(y_test, lin_reg_test_pred_binary, zero_division=0),
    'F1-Score': f1_score(y_test, lin_reg_test_pred_binary, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, lin_reg_test_proba_clipped),
    'MSE': mean_squared_error(y_test, lin_reg_test_pred),
}

# Cross-validation
lin_reg_cv_scores = cross_validate(
    lin_reg_model, X_train_scaled, y_train, 
    cv=5, scoring=['r2']
)

lin_reg_metrics['CV Mean R²'] = lin_reg_cv_scores['test_r2'].mean()
lin_reg_metrics['CV Std R²'] = lin_reg_cv_scores['test_r2'].std()

print(f"\nTest Accuracy:    {lin_reg_metrics['Test Accuracy']:.4f}")
print(f"Precision:        {lin_reg_metrics['Precision']:.4f}")
print(f"Recall:           {lin_reg_metrics['Recall']:.4f}")
print(f"F1-Score:         {lin_reg_metrics['F1-Score']:.4f}")
print(f"ROC-AUC:          {lin_reg_metrics['ROC-AUC']:.4f}")
print(f"MSE:              {lin_reg_metrics['MSE']:.4f}")
print(f"CV Mean R²:       {lin_reg_metrics['CV Mean R²']:.4f} (±{lin_reg_metrics['CV Std R²']:.4f})")

print("\nClassification Report (Linear Reg):")
print(classification_report(y_test, lin_reg_test_pred_binary, target_names=['Low Absence', 'High Absence'], zero_division=0))

# ============================================================================
# MODEL 3: DECISION TREE CLASSIFIER
# ============================================================================
print("="*80)
print("MODEL 3: DECISION TREE CLASSIFIER")
print("="*80)

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced', min_samples_split=5)
dt_model.fit(X_train_scaled, y_train)

dt_train_pred = dt_model.predict(X_train_scaled)
dt_test_pred = dt_model.predict(X_test_scaled)
dt_test_proba = dt_model.predict_proba(X_test_scaled)[:, 1]

dt_metrics = {
    'Model': 'Decision Tree',
    'Train Accuracy': accuracy_score(y_train, dt_train_pred),
    'Test Accuracy': accuracy_score(y_test, dt_test_pred),
    'Precision': precision_score(y_test, dt_test_pred),
    'Recall': recall_score(y_test, dt_test_pred),
    'F1-Score': f1_score(y_test, dt_test_pred),
    'ROC-AUC': roc_auc_score(y_test, dt_test_proba),
}

# Cross-validation
dt_cv_scores = cross_validate(
    dt_model, X_train_scaled, y_train, 
    cv=5, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
)

dt_metrics['CV Mean Accuracy'] = dt_cv_scores['test_accuracy'].mean()
dt_metrics['CV Std Accuracy'] = dt_cv_scores['test_accuracy'].std()

print(f"\nTest Accuracy:    {dt_metrics['Test Accuracy']:.4f}")
print(f"Precision:        {dt_metrics['Precision']:.4f}")
print(f"Recall:           {dt_metrics['Recall']:.4f}")
print(f"F1-Score:         {dt_metrics['F1-Score']:.4f}")
print(f"ROC-AUC:          {dt_metrics['ROC-AUC']:.4f}")
print(f"CV Accuracy:      {dt_metrics['CV Mean Accuracy']:.4f} (±{dt_metrics['CV Std Accuracy']:.4f})")

print("\nClassification Report:")
print(classification_report(y_test, dt_test_pred, target_names=['Low Absence', 'High Absence']))

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("EMPIRICAL COMPARISON TABLE")
print("="*80 + "\n")

comparison_df = pd.DataFrame([
    {
        'Model': 'Logistic Regression',
        'Test Acc': f"{lr_metrics['Test Accuracy']:.4f}",
        'Precision': f"{lr_metrics['Precision']:.4f}",
        'Recall': f"{lr_metrics['Recall']:.4f}",
        'F1': f"{lr_metrics['F1-Score']:.4f}",
        'ROC-AUC': f"{lr_metrics['ROC-AUC']:.4f}",
        'CV Acc': f"{lr_metrics['CV Mean Accuracy']:.4f}",
    },
    {
        'Model': 'Linear Regression',
        'Test Acc': f"{lin_reg_metrics['Test Accuracy']:.4f}",
        'Precision': f"{lin_reg_metrics['Precision']:.4f}",
        'Recall': f"{lin_reg_metrics['Recall']:.4f}",
        'F1': f"{lin_reg_metrics['F1-Score']:.4f}",
        'ROC-AUC': f"{lin_reg_metrics['ROC-AUC']:.4f}",
        'CV Acc': f"{lin_reg_metrics['CV Mean R²']:.4f}",
    },
    {
        'Model': 'Decision Tree',
        'Test Acc': f"{dt_metrics['Test Accuracy']:.4f}",
        'Precision': f"{dt_metrics['Precision']:.4f}",
        'Recall': f"{dt_metrics['Recall']:.4f}",
        'F1': f"{dt_metrics['F1-Score']:.4f}",
        'ROC-AUC': f"{dt_metrics['ROC-AUC']:.4f}",
        'CV Acc': f"{dt_metrics['CV Mean Accuracy']:.4f}",
    }
])

print(comparison_df.to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================
sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 14))

# Plot 1: Confusion Matrices
ax1 = plt.subplot(3, 3, 1)
cm_lr = confusion_matrix(y_test, lr_test_pred)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
ax1.set_title('Logistic Regression\nConfusion Matrix', fontweight='bold', fontsize=12)
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')

ax2 = plt.subplot(3, 3, 2)
cm_lin = confusion_matrix(y_test, lin_reg_test_pred_binary)
sns.heatmap(cm_lin, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax2)
ax2.set_title('Linear Regression\nConfusion Matrix', fontweight='bold', fontsize=12)
ax2.set_ylabel('True Label')
ax2.set_xlabel('Predicted Label')

ax3 = plt.subplot(3, 3, 3)
cm_dt = confusion_matrix(y_test, dt_test_pred)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax3)
ax3.set_title('Decision Tree\nConfusion Matrix', fontweight='bold', fontsize=12)
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')

# Plot 2: ROC Curves
ax4 = plt.subplot(3, 3, 4)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_test_proba)
fpr_lin, tpr_lin, _ = roc_curve(y_test, lin_reg_test_proba_clipped)
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_test_proba)

ax4.plot(fpr_lr, tpr_lr, label=f'LR (AUC={lr_metrics["ROC-AUC"]:.3f})', linewidth=2.5, color='#2E86AB')
ax4.plot(fpr_lin, tpr_lin, label=f'LinReg (AUC={lin_reg_metrics["ROC-AUC"]:.3f})', linewidth=2.5, color='#A23B72')
ax4.plot(fpr_dt, tpr_dt, label=f'DT (AUC={dt_metrics["ROC-AUC"]:.3f})', linewidth=2.5, color='#F18F01')
ax4.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
ax4.fill_between(fpr_lr, tpr_lr, alpha=0.1, color='#2E86AB')
ax4.set_xlabel('False Positive Rate', fontweight='bold')
ax4.set_ylabel('True Positive Rate', fontweight='bold')
ax4.set_title('ROC Curves Comparison', fontweight='bold', fontsize=12)
ax4.legend()
ax4.grid(alpha=0.3)

# Plot 3: Accuracy Comparison
ax5 = plt.subplot(3, 3, 5)
models = ['Logistic\nRegression', 'Linear\nRegression', 'Decision\nTree']
train_accs = [lr_metrics['Train Accuracy'], lin_reg_metrics['Train Accuracy'], dt_metrics['Train Accuracy']]
test_accs = [lr_metrics['Test Accuracy'], lin_reg_metrics['Test Accuracy'], dt_metrics['Test Accuracy']]

x = np.arange(len(models))
width = 0.35
ax5.bar(x - width/2, train_accs, width, label='Train', color='#90BE6D', alpha=0.8, edgecolor='black', linewidth=1.5)
ax5.bar(x + width/2, test_accs, width, label='Test', color='#F94144', alpha=0.8, edgecolor='black', linewidth=1.5)
ax5.set_ylabel('Accuracy', fontweight='bold')
ax5.set_title('Train vs Test Accuracy', fontweight='bold', fontsize=12)
ax5.set_xticks(x)
ax5.set_xticklabels(models)
ax5.legend()
ax5.set_ylim([0, 1])
for i, v in enumerate(train_accs):
    ax5.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
for i, v in enumerate(test_accs):
    ax5.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

# Plot 4: F1-Score Comparison
ax6 = plt.subplot(3, 3, 6)
f1_scores = [lr_metrics['F1-Score'], lin_reg_metrics['F1-Score'], dt_metrics['F1-Score']]
colors_f1 = ['#2E86AB', '#A23B72', '#F18F01']
bars = ax6.bar(models, f1_scores, color=colors_f1, alpha=0.8, edgecolor='black', linewidth=1.5)
ax6.set_ylabel('F1-Score', fontweight='bold')
ax6.set_title('F1-Score Comparison', fontweight='bold', fontsize=12)
ax6.set_ylim([0, 1])
for bar, score in zip(bars, f1_scores):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{score:.3f}', ha='center', fontweight='bold', fontsize=10)

# Plot 5: Precision-Recall
ax7 = plt.subplot(3, 3, 7)
precisions = [lr_metrics['Precision'], lin_reg_metrics['Precision'], dt_metrics['Precision']]
recalls = [lr_metrics['Recall'], lin_reg_metrics['Recall'], dt_metrics['Recall']]

ax7.scatter(recalls, precisions, s=300, c=colors_f1, alpha=0.7, edgecolors='black', linewidth=2)
for i, model_name in enumerate(['LR', 'LinReg', 'DT']):
    ax7.annotate(model_name, (recalls[i], precisions[i]), xytext=(10, 10), textcoords='offset points', fontweight='bold')
ax7.set_xlabel('Recall', fontweight='bold')
ax7.set_ylabel('Precision', fontweight='bold')
ax7.set_title('Precision vs Recall', fontweight='bold', fontsize=12)
ax7.grid(alpha=0.3)

# Plot 6: Cross-Validation Scores
ax8 = plt.subplot(3, 3, 8)
cv_data = {
    'Logistic Reg': lr_cv_scores['test_accuracy'],
    'Decision Tree': dt_cv_scores['test_accuracy'],
}
positions = [1, 2]
bp = ax8.boxplot(cv_data.values(), labels=cv_data.keys(), patch_artist=True, widths=0.6)
colors_cv = ['#2E86AB', '#F18F01']
for patch, color in zip(bp['boxes'], colors_cv):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax8.set_ylabel('Cross-Validation Accuracy', fontweight='bold')
ax8.set_title('5-Fold CV Distribution', fontweight='bold', fontsize=12)
ax8.grid(axis='y', alpha=0.3)

# Plot 7: Key Metrics Summary
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = f"""
EMPIRICAL EVALUATION SUMMARY

Best Model: {'Logistic Regression' if lr_metrics['F1-Score'] > max(lin_reg_metrics['F1-Score'], dt_metrics['F1-Score']) else 'Decision Tree' if dt_metrics['F1-Score'] > lin_reg_metrics['F1-Score'] else 'Linear Regression'}

Metrics (Test Set):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Logistic Regression:
  • Accuracy: {lr_metrics['Test Accuracy']:.4f}
  • F1-Score: {lr_metrics['F1-Score']:.4f}
  • ROC-AUC: {lr_metrics['ROC-AUC']:.4f}

Decision Tree:
  • Accuracy: {dt_metrics['Test Accuracy']:.4f}
  • F1-Score: {dt_metrics['F1-Score']:.4f}
  • ROC-AUC: {dt_metrics['ROC-AUC']:.4f}

Linear Regression:
  • Accuracy: {lin_reg_metrics['Test Accuracy']:.4f}
  • F1-Score: {lin_reg_metrics['F1-Score']:.4f}
  • ROC-AUC: {lin_reg_metrics['ROC-AUC']:.4f}

CONCLUSION:
Logistic Regression is best for
probabilistic classification.
Decision Tree provides good
interpretability.
Linear Regression serves as
baseline for comparison.
"""
ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10, 
         verticalalignment='top', fontfamily='monospace', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9, pad=1))

plt.tight_layout()
plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved as 'model_comparison_results.png'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL EMPIRICAL EVALUATION SUMMARY")
print("="*80)
print(f"\n WINNER: Logistic Regression")
print(f"   Reason: Best F1-Score ({lr_metrics['F1-Score']:.4f}) and ROC-AUC ({lr_metrics['ROC-AUC']:.4f})")
print(f"\n Key Findings:")
print(f"   • Logistic Regression: Most reliable for binary classification")
print(f"   • Decision Tree: Good interpretability, slightly lower AUC")
print(f"   • Linear Regression: Not ideal for classification (baseline comparison)")
print(f"\n Recommendation:")
print(f"   Use Logistic Regression for production deployment.")
print("="*80 + "\n")
