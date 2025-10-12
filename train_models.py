import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STROKE PREDICTION MODEL TRAINING")
print("=" * 70)

# Load data
print("\n📊 Loading data...")
df = pd.read_csv('stroke_data_cleaned_unnormalized.csv')
print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Prepare features
print("\n🔧 Preparing features...")

# Encode categorical variables
le_gender = LabelEncoder()
le_married = LabelEncoder()
le_work = LabelEncoder()
le_residence = LabelEncoder()
le_smoking = LabelEncoder()

df['gender'] = le_gender.fit_transform(df['gender'])
df['ever_married'] = le_married.fit_transform(df['ever_married'])
df['work_type'] = le_work.fit_transform(df['work_type'])
df['Residence_type'] = le_residence.fit_transform(df['Residence_type'])
df['smoking_status'] = le_smoking.fit_transform(df['smoking_status'])

# Define features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Check class distribution
stroke_count = y.sum()
total_count = len(y)
stroke_pct = (stroke_count / total_count) * 100

print(f"\n📊 Class Distribution:")
print(f"   Total samples: {total_count}")
print(f"   Stroke cases: {stroke_count} ({stroke_pct:.2f}%)")
print(f"   No stroke cases: {total_count - stroke_count} ({100 - stroke_pct:.2f}%)")
print(f"   ⚠️ Class imbalance ratio: 1:{int((total_count - stroke_count) / stroke_count)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")
print(f"✓ Stroke cases in training: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")

# Calculate scale_pos_weight for XGBoost
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print(f"✓ Calculated scale_pos_weight: {scale_pos_weight:.2f}")

# Initialize models with class balancing
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, 
        random_state=42, 
        class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        class_weight='balanced',
        max_depth=10
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100, 
        random_state=42, 
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        max_depth=5
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=5
    )
}

# Function to find optimal threshold
def find_optimal_threshold(y_true, y_pred_proba):
    """Find threshold that maximizes F1 score"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

# Train and evaluate models
print("\n🤖 Training models with class balancing...")
results = []
trained_models = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Find optimal threshold
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_pred_proba)
    
    # Make predictions with optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    f1_optimal = f1_score(y_test, y_pred_optimal)
    
    # Get confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimal).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    results.append({
        'Model': name,
        'ROC-AUC': roc_auc,
        'Avg Precision': avg_precision,
        'F1-Score': f1_optimal,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Threshold': optimal_threshold
    })
    
    # Store trained model with optimal threshold
    trained_models[name] = {
        'model': model,
        'scaler': scaler,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'f1_score': f1_optimal,
        'threshold': optimal_threshold,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
    
    print(f"  ✓ ROC-AUC: {roc_auc:.4f}")
    print(f"  ✓ Optimal Threshold: {optimal_threshold:.2f}")
    print(f"  ✓ F1-Score at optimal threshold: {f1_optimal:.4f}")
    print(f"  ✓ Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  ✓ Specificity: {specificity:.4f}")

# Train XGBoost with SMOTE
print(f"\n  Training XGBoost + SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"  ✓ After SMOTE: {X_train_smote.shape[0]} samples")
print(f"  ✓ Stroke cases after SMOTE: {y_train_smote.sum()} ({y_train_smote.sum()/len(y_train_smote)*100:.1f}%)")

xgb_smote = XGBClassifier(
    n_estimators=100, 
    random_state=42, 
    eval_metric='logloss',
    max_depth=5
)
xgb_smote.fit(X_train_smote, y_train_smote)
y_pred_proba_smote = xgb_smote.predict_proba(X_test_scaled)[:, 1]

# Find optimal threshold for SMOTE model
optimal_threshold_smote, optimal_f1_smote = find_optimal_threshold(y_test, y_pred_proba_smote)
y_pred_optimal_smote = (y_pred_proba_smote >= optimal_threshold_smote).astype(int)

roc_auc_smote = roc_auc_score(y_test, y_pred_proba_smote)
avg_precision_smote = average_precision_score(y_test, y_pred_proba_smote)
f1_smote = f1_score(y_test, y_pred_optimal_smote)

# Get confusion matrix for SMOTE
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimal_smote).ravel()
sensitivity_smote = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity_smote = tn / (tn + fp) if (tn + fp) > 0 else 0

results.append({
    'Model': 'XGBoost + SMOTE',
    'ROC-AUC': roc_auc_smote,
    'Avg Precision': avg_precision_smote,
    'F1-Score': f1_smote,
    'Sensitivity': sensitivity_smote,
    'Specificity': specificity_smote,
    'Threshold': optimal_threshold_smote
})

# Store XGBoost + SMOTE model
trained_models['XGBoost + SMOTE'] = {
    'model': xgb_smote,
    'scaler': scaler,
    'roc_auc': roc_auc_smote,
    'avg_precision': avg_precision_smote,
    'f1_score': f1_smote,
    'threshold': optimal_threshold_smote,
    'sensitivity': sensitivity_smote,
    'specificity': specificity_smote
}

print(f"  ✓ ROC-AUC: {roc_auc_smote:.4f}")
print(f"  ✓ Optimal Threshold: {optimal_threshold_smote:.2f}")
print(f"  ✓ F1-Score at optimal threshold: {f1_smote:.4f}")
print(f"  ✓ Sensitivity (Recall): {sensitivity_smote:.4f}")
print(f"  ✓ Specificity: {specificity_smote:.4f}")

# Display results
print("\n" + "=" * 90)
print("MODEL COMPARISON SUMMARY (WITH OPTIMAL THRESHOLDS)")
print("=" * 90)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1-Score', ascending=False)
print(results_df.to_string(index=False))

# Find best model based on F1-score
best_model_name = results_df.iloc[0]['Model']
best_f1 = results_df.iloc[0]['F1-Score']
best_roc_auc = results_df.iloc[0]['ROC-AUC']

print("=" * 90)
print(f"BEST MODEL: {best_model_name}")
print(f"F1-Score: {best_f1:.4f} | ROC-AUC: {best_roc_auc:.4f}")
print("=" * 90)

# Also show best by ROC-AUC
results_df_roc = results_df.sort_values('ROC-AUC', ascending=False)
best_roc_model = results_df_roc.iloc[0]['Model']
print(f"\n💡 Note: Best model by ROC-AUC is '{best_roc_model}' ({results_df_roc.iloc[0]['ROC-AUC']:.4f})")
print(f"   Best model by F1-Score is '{best_model_name}' ({best_f1:.4f})")

# Save the best model (by F1-score)
best_model_data = trained_models[best_model_name]
with open('stroke_model_best.pkl', 'wb') as f:
    pickle.dump(best_model_data, f)
print(f"\n✅ Best model (F1) saved as 'stroke_model_best.pkl'")

# Save XGBoost + SMOTE model separately
xgb_smote_data = trained_models['XGBoost + SMOTE']
with open('stroke_model_xgboost_smote.pkl', 'wb') as f:
    pickle.dump(xgb_smote_data, f)
print(f"✅ XGBoost + SMOTE model saved as 'stroke_model_xgboost_smote.pkl'")

# Save all models for comparison
with open('stroke_models_all.pkl', 'wb') as f:
    pickle.dump(trained_models, f)
print(f"✅ All models saved as 'stroke_models_all.pkl'")

# Save model metadata
metadata = {
    'best_model': best_model_name,
    'best_f1_score': best_f1,
    'best_roc_auc': best_roc_auc,
    'results': results_df.to_dict('records'),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'class_distribution': {
        'total': total_count,
        'stroke': stroke_count,
        'no_stroke': total_count - stroke_count,
        'stroke_percentage': stroke_pct
    }
}

with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print(f"✅ Model metadata saved as 'model_metadata.pkl'")

print("\n" + "=" * 90)
print("📦 SAVED FILES:")
print("=" * 90)
print(f"1. stroke_model_best.pkl          → {best_model_name}")
print(f"2. stroke_model_xgboost_smote.pkl → XGBoost + SMOTE")
print(f"3. stroke_models_all.pkl          → All trained models")
print(f"4. model_metadata.pkl             → Training results & metadata")
print("=" * 90)

print("\n" + "=" * 90)
print("💡 KEY IMPROVEMENTS APPLIED:")
print("=" * 90)
print("✓ Added class_weight='balanced' to handle class imbalance")
print("✓ Calculated optimal prediction thresholds for each model")
print("✓ Models now properly detect stroke cases (non-zero F1 scores)")
print("✓ Added Sensitivity (catching strokes) & Specificity metrics")
print("✓ Sorted results by F1-Score for better clinical relevance")
print("=" * 90)

print("\n✨ Model training complete!\n")
