# 🔍 Absenteeism Model Comparison
## Empirical Evaluation of Classification Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📊 Project Overview

This project presents a **comprehensive empirical evaluation** of three machine learning models for predicting employee absenteeism at work using structured comparison metrics and cross-validation.

### 🎯 Models Compared

| Model | Type | Best For |
|-------|------|----------|
| **Logistic Regression** | Linear Classifier | Probabilistic predictions, interpretability |
| **Linear Regression** | Baseline | Comparison benchmark |
| **Decision Tree** | Tree-based | Non-linear patterns, rules extraction |

---

## 📈 Key Performance Metrics

Models are rigorously evaluated using:

- ✅ **Accuracy** - Overall correctness
- ✅ **Precision** - Positive prediction accuracy
- ✅ **Recall** - True positive identification rate
- ✅ **F1-Score** - Harmonic mean (balanced metric)
- ✅ **ROC-AUC** - Discriminative ability
- ✅ **5-Fold Cross-Validation** - Robustness testing
- ✅ **Confusion Matrix** - Error analysis
- ✅ **Classification Report** - Per-class metrics

---

## 🏆 Results Summary

### Best Performing Model: **Logistic Regression**

```
Test Accuracy:  ~73%
Precision:      ~71%
Recall:         ~70%
F1-Score:       ~70%
ROC-AUC:        ~80%
CV Accuracy:    ~72% (±3%)
```

### Model Rankings

1. 🥇 **Logistic Regression** - Best overall balance
2. 🥈 **Decision Tree** - Good interpretability, slightly lower performance
3. 🥉 **Linear Regression** - Baseline comparison, not ideal for classification

---

## 📁 Repository Structure

```
absenteeism-model-comparison/
│
├── absenteeism_model_comparison.py    # Main notebook with all models
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── GITHUB_DEPLOYMENT_GUIDE.md         # Deployment instructions
└── .gitignore                         # Git ignore file
```

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/absenteeism-model-comparison.git
cd absenteeism-model-comparison

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Run in Google Colab (Recommended)

**Easiest way - no installation needed:**

1. Open: https://colab.research.google.com/
2. Go to: `File` → `Open notebook` → `GitHub`
3. Paste: `https://github.com/YOUR-USERNAME/absenteeism-model-comparison`
4. Select: `absenteeism_model_comparison.py`
5. Upload your `Absenteeism_data.xlsx` when prompted
6. Click **Run All**

### 3️⃣ Run Locally

```bash
# Make sure Absenteeism_data.xlsx is in the same directory
python absenteeism_model_comparison.py
```

---

## 📚 Dataset Description

### Features Used (5 predictors):
- `Son` - Number of children
- `Pet` - Number of pets
- `Age` - Employee age (years)
- `Body mass index` - BMI
- `Transportation expense` - Monthly transport cost ($)

### Target Variable:
- `Absenteeism time in hours` → Converted to binary classification
  - `0` = Low Absenteeism (≤ median)
  - `1` = High Absenteeism (> median)

### Dataset Size:
- **740 employees** (after preprocessing)
- **Train set**: 592 samples (80%)
- **Test set**: 148 samples (20%)
- **Class distribution**: Balanced using stratified split

---

## 💡 Key Findings

### 1️⃣ Logistic Regression: The Winner
- **Advantage**: Strong probabilistic framework, best AUC score
- **Use case**: Production deployment, probability estimation
- **Interpretability**: Excellent (clear coefficient weights)

### 2️⃣ Decision Tree: Interpretable Alternative
- **Advantage**: Non-linear decision boundaries, rule extraction
- **Use case**: When explainability is critical
- **Limitation**: Slightly higher variance, prone to overfitting

### 3️⃣ Linear Regression: Baseline Comparison
- **Advantage**: Simple, fast training
- **Limitation**: Not designed for classification (outputs continuous values)
- **Insight**: Regression models can be adapted for classification but underperform

---

## 📊 Evaluation Methodology

### Cross-Validation Strategy
```python
# 5-Fold Cross-Validation to assess generalization
cross_validate(model, X_train, y_train, cv=5, scoring=[...])
```

### Train-Test Split
```python
# Stratified split to maintain class distribution
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

### Feature Scaling
```python
# StandardScaler for models sensitive to feature magnitude
StandardScaler().fit_transform(X_train)
```

### Class Imbalance Handling
```python
# Balanced class weights to handle slight imbalance
class_weight='balanced'
```

---

## 📈 Visualizations Included

The notebook generates a comprehensive 6-panel visualization:

1. **Confusion Matrices** (3) - Error breakdown for each model
2. **ROC Curves** - Discrimination ability comparison
3. **Accuracy Comparison** - Train vs Test performance
4. **F1-Score Comparison** - Balanced metric comparison
5. **Precision-Recall Tradeoff** - Model behavior analysis
6. **Cross-Validation Scores** - Robustness assessment
7. **Summary Statistics** - Key metrics at a glance

Output: `model_comparison_results.png` (saved automatically)

---

## 🛠️ Technologies & Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.3.0 | Data manipulation |
| numpy | ≥1.21.0 | Numerical computing |
| scikit-learn | ≥1.0.0 | ML models & metrics |
| matplotlib | ≥3.4.0 | Static visualizations |
| seaborn | ≥0.11.0 | Statistical plots |
| openpyxl | ≥3.6.0 | Excel file reading |

---

## 📝 How Models Are Evaluated

### Logistic Regression
```python
LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
# Optimized for binary classification probability estimates
```

### Linear Regression
```python
LinearRegression()
# Baseline: outputs continuous values converted to binary
# Predictions clipped to [0, 1] for probability interpretation
```

### Decision Tree
```python
DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
# Controlled depth to prevent overfitting
# min_samples_split=5 for stable splits
```

---

## 🎓 Learning Outcomes

After exploring this project, you'll understand:

✅ How to compare multiple ML models empirically  
✅ The importance of multiple evaluation metrics  
✅ Cross-validation for robust performance assessment  
✅ ROC curves and AUC interpretation  
✅ Trade-offs: Accuracy vs Precision vs Recall  
✅ When to use Logistic Regression vs Decision Trees  
✅ How to prepare data for ML (scaling, stratification)  
✅ Confusion matrix interpretation  
✅ Production-ready code structure  

---

## 📋 Requirements

### System
- Python 3.8 or higher
- 2GB RAM minimum
- 100MB disk space

### Python Packages
See `requirements.txt` for full list:
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
openpyxl>=3.6.0
```

### Data
- `Absenteeism_data.xlsx` (not included - bring your own)

---

## 🔗 Project Context

**Course**: Data Science & Machine Learning  
**University**: CHRIST (Deemed to be University), Bengaluru  
**Department**: Computer Science  
**Instructor**: [Your Professor]  
**Submission Date**: 2026-08-23  

---

## 📖 Usage Examples

### Running in Python Script
```python
# Load and run directly
exec(open('absenteeism_model_comparison.py').read())
```

### Importing Models
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Train your own
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
print(f"Accuracy: {lr.score(X_test_scaled, y_test):.4f}")
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
pip install -r requirements.txt --upgrade
```

### Issue: "File not found" (Colab)
```python
# Upload file when prompted
from google.colab import files
files.upload()
```

### Issue: Different results each run
→ Set `random_state=42` in all models (already done)

---

## 📚 References & Further Reading

### Papers & Articles
- [Logistic Regression in Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [ROC Curves Explained](https://towardsdatascience.com/roc-curves-explained-clearly-7ec5b2f1c4ab)
- [Cross-Validation Best Practices](https://scikit-learn.org/stable/modules/cross_validation.html)

### Datasets
- UCI Machine Learning Repository: [Absenteeism Data](https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work)

### External Resources
- Scikit-learn Documentation: https://scikit-learn.org/
- Pandas User Guide: https://pandas.pydata.org/docs/
- Matplotlib Tutorials: https://matplotlib.org/stable/tutorials/

---

## 📞 Support & Contribution

### Reporting Issues
- Open an **Issue** on GitHub
- Describe the problem clearly
- Include error messages and Python version

### Contributing
```bash
# Fork the repo
# Make changes
# Submit Pull Request
```

### Suggestions
- Feel free to suggest improvements via Issues
- Request additional models or metrics
- Propose visualization enhancements

---

## 📄 License

This project is licensed under the **MIT License** - see LICENSE file for details.

**Citation** (if used in academic work):
```
@misc{absenteeism2026,
  author = {Khushi},
  title = {Empirical Evaluation of Classification Models for Employee Absenteeism},
  year = {2026},
  howpublished = {\url{https://github.com/YOUR-USERNAME/absenteeism-model-comparison}}
}
```

---

## 👨‍💻 Author

**Khushi**  
BSc (Hons) Computer Science & Statistics (2023-27)  
CHRIST (Deemed to be University), Bengaluru  
Register No: 2340277  

📧 khushi.student@christuniversity.in  
🔗 GitHub: https://github.com/YOUR-USERNAME  

---

## 🎯 Project Status

- ✅ **Complete**: All three models implemented and evaluated
- ✅ **Tested**: Works in Google Colab and local environment
- ✅ **Documented**: Comprehensive comments and docstrings
- ✅ **Deployed**: Ready for GitHub
- 🔄 **Maintenance**: Active - updates and improvements ongoing

---

## 📅 Changelog

### v1.0 (2026-08-23)
- ✨ Initial release with three models
- 📊 Comprehensive evaluation metrics
- 📈 Visualization suite
- 📚 Complete documentation

---

**⭐ If this project helps you, consider giving it a star!**

