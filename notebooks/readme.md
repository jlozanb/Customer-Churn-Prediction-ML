# Notebooks: Churn_Prediction_Analysis.ipynb

Walkthrough of what each section does, why it's there, and what it
concludes. The notebook is already executed with its outputs and charts
saved, so it can be read directly on GitHub without needing to run it.

## 1. Business question and approach
States the churn definition, why it matters for the business, and the
plan: an interpretable baseline model plus a stronger model, evaluated
with imbalance-aware metrics, ending in retention recommendations rather
than a bare prediction score.

## 2. Load and inspect the data
Basic checks: shape, dtypes, missing values. Confirms the only missing
column is `avg_satisfaction_score`, and that it's missing specifically for
customers who never contacted support, which is expected and meaningful.

## 3. Exploratory data analysis
- Checks the class balance first (17.8% churn), since this determines
  which evaluation metrics will actually be meaningful later
- Boxplots of the numeric features split by churn status, to get a first
  read on which features separate the two groups
- A correlation matrix as a quick, if limited, second check

## 4. Feature engineering
- Creates an explicit `ever_contacted_support` flag before filling the
  missing satisfaction scores, so "never contacted support" isn't lost as
  information
- One-hot encodes `country` and `device`
- Splits into train/test sets (75/25), stratified on the churn label so
  both sets keep the same churn rate
- Notes the data leakage consideration: in a production version, feature
  timing relative to the churn event would need to be checked carefully

## 5. Logistic Regression (baseline)
Trains a logistic regression with scaled features and `class_weight="balanced"`.
Reports precision, recall, F1 and ROC-AUC, then plots the coefficients to
check that the model's story matches the EDA (it does: recency and
support tickets increase churn risk, tenure and engagement decrease it).

## 6. Random Forest
Trains a Random Forest with fixed, reasonable-looking hyperparameters
(not tuned), also with `class_weight="balanced"`.

## 7. Comparing the two models
Plots ROC curves and confusion matrices for both models side by side. The
notebook is explicit that the untuned Random Forest does not beat the
simpler Logistic Regression on ROC-AUC in this case (0.715 vs 0.726),
and treats that as a real finding rather than smoothing it over. Also
includes a precision-recall curve, since the right operating threshold
depends on the real-world cost of a false positive (an unnecessary
retention offer) versus a false negative (a lost customer).

## 8. Feature importance and retention insights
Extracts Random Forest feature importance and turns the ranking into three
concrete retention actions (recency-triggered re-engagement, treating
support tickets as a retention signal, focusing on newer/low-engagement
customers), rather than stopping at the chart.

## 9. Limitations
Explicit list of what a more rigorous version would add: a time-based
train/test split instead of a random one, hyperparameter tuning with
cross-validation, permutation importance or SHAP instead of the built-in
feature importances, explicit resampling techniques like SMOTE, and
sizing the business impact of the recommendations rather than leaving them
qualitative.

## Key libraries used and why
- `sklearn.model_selection.train_test_split` with `stratify=y` to keep the
  churn rate consistent across train and test
- `sklearn.preprocessing.StandardScaler` for the logistic regression
  features (tree-based models don't need this, logistic regression does)
- `sklearn.linear_model.LogisticRegression` and
  `sklearn.ensemble.RandomForestClassifier`, both with
  `class_weight="balanced"` to account for the churn imbalance
- `sklearn.metrics` (classification_report, roc_auc_score, roc_curve,
  precision_recall_curve, confusion_matrix) for imbalance-aware evaluation
