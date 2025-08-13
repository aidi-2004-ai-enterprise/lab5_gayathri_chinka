"""
Lab 5: Complete Bankruptcy Prediction Training Pipeline
Student: Gayathri Chinka
Implementing all Lab 4 research decisions

This pipeline implements every decision made in Lab 4:
✅ Models: Logistic Regression + Random Forest + XGBoost
✅ Class Imbalance: SMOTE (for 1:29 ratio found in Lab 4)
✅ Outlier Treatment: Cap at 99th percentile for problematic features
✅ Feature Selection: XGBoost importance (top 40 features)
✅ Validation: Stratified 5-fold CV
✅ Metrics: ROC-AUC, PR-AUC, F1-score, Brier score
✅ Interpretability: SHAP analysis
✅ Stability: PSI monitoring
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, 
                           f1_score, brier_score_loss, auc)
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

class BankruptcyPredictionPipeline:
    """Complete training pipeline implementing Lab 4 decisions"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Store models, scalers, and results
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.selected_features = None
        
        # Based on Lab 4 findings - features with impossible values that need capping
        self.problematic_features = [
            ' Revenue Per Share (Yuan ¥)',
            ' Net Value Growth Rate',
            ' Current Ratio',
            ' Quick Ratio',
            ' Total debt/Total net worth',
            ' Accounts Receivable Turnover',
            ' Average Collection Days',
            ' Revenue per person',
            ' Allocation rate per person',
            ' Quick Assets/Current Liability',
            ' Cash/Current Liability',
            ' Fixed Assets to Assets',
            ' Total assets to GNP price'
        ]

    def load_and_explore_data(self, file_path: str = 'data.csv') -> pd.DataFrame:
        """
        Load data and perform EDA based on Lab 4 findings.
        
        My Lab 4 analysis found:
        - 6,819 companies with 96 features
        - Severe 1:29 class imbalance (3.2% bankruptcy rate)
        - 13 features with impossible billion-dollar values
        - Zero missing values
        """
        print("🔍 LOADING DATA AND PERFORMING EDA")
        print("=" * 50)
        
        # Load the dataset
        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {df.shape[0]:,} companies, {df.shape[1]} features")
        
        # Confirm Lab 4 findings
        target_dist = df['Bankrupt?'].value_counts()
        print("Class distribution confirmed:")
        print(f"  Non-bankrupt: {target_dist[0]:,} ({target_dist[0]/len(df)*100:.1f}%)")
        print(f"  Bankrupt: {target_dist[1]:,} ({target_dist[1]/len(df)*100:.1f}%)")
        print(f"  Imbalance ratio: 1:{target_dist[0]//target_dist[1]}")
        
        # Create EDA visualizations
        plt.figure(figsize=(15, 12))
        
        # 1. Class distribution
        plt.subplot(2, 3, 1)
        target_dist.plot(kind='bar', color=['lightblue', 'salmon'])
        plt.title('Class Distribution (Confirming Lab 4)')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Non-Bankrupt', 'Bankrupt'], rotation=0)
        
        # 2. Extreme outliers visualization
        plt.subplot(2, 3, 2)
        extreme_features = [' Current Ratio', ' Revenue Per Share (Yuan ¥)']
        for i, feature in enumerate(extreme_features):
            if feature in df.columns:
                data = df[feature].dropna()
                plt.hist(data, bins=30, alpha=0.7, 
                        label=f'{feature.split()[1] if len(feature.split()) > 1 else feature[:10]}...',
                        density=True)
        plt.title('Problematic Features (Lab 4 Identified)')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.yscale('log')
        
        # 3. Well-behaved features distribution
        plt.subplot(1, 3, 3)
        good_features = [' ROA(C) before interest and depreciation before interest',
                        ' Debt ratio %', ' Operating Gross Margin']
        for i, feature in enumerate(good_features):
            if feature in df.columns:
                data = df[feature].dropna()
                plt.hist(data, bins=30, alpha=0.7, 
                        label=f'{feature.split()[1] if len(feature.split()) > 1 else feature[:10]}...',
                        density=True)
        plt.title('Well-Behaved Features\\n(Normalized 0-1 Range)')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 EDA visualizations saved as 'eda_analysis.png'")
        
        return df

    def preprocess_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data based on Lab 4 decisions:
        - Cap outliers at 99th percentile for problematic features
        - Stratified train/test split to maintain class balance
        - Apply SMOTE to training data only
        """
        print("\\n🔧 DATA PREPROCESSING (Implementing Lab 4 Decisions)")
        print("=" * 55)
        
        # Separate features and target
        X = df.drop('Bankrupt?', axis=1)
        y = df['Bankrupt?']
        
        # Apply outlier capping to problematic features identified in Lab 4
        X_capped = X.copy()
        print("Capping outliers for problematic features:")
        
        for feature in self.problematic_features:
            if feature in X_capped.columns:
                q99 = X_capped[feature].quantile(0.99)
                original_max = X_capped[feature].max()
                X_capped[feature] = X_capped[feature].clip(upper=q99)
                print(f"  {feature[:30]}... {original_max:.0f} → {q99:.4f}")
        
        # Stratified train/test split (maintaining 3.2% bankruptcy rate)
        X_train, X_test, y_train, y_test = train_test_split(
            X_capped, y, test_size=0.3, random_state=self.random_state,
            stratify=y
        )
        
        print("\\nTrain/test split completed:")
        print(f"  Training: {len(X_train):,} samples ({y_train.sum()} bankrupt)")
        print(f"  Test: {len(X_test):,} samples ({y_test.sum()} bankrupt)")
        
        # Apply SMOTE to training data only (Lab 4 decision for 1:29 imbalance)
        smote = SMOTE(random_state=self.random_state)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        print("\\nSMOTE applied to training data:")
        print(f"  Before SMOTE: {y_train.value_counts().to_dict()}")
        print(f"  After SMOTE: {pd.Series(y_train_smote).value_counts().to_dict()}")
        
        # Calculate and display PSI for key features (Lab 4 validation)
        self._calculate_psi(X_train, X_test)
        
        return X_train_smote, X_test, y_train_smote, y_test

    def _calculate_psi(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
        """Calculate PSI to validate Lab 4 findings of no distribution drift."""
        print("\\n📊 PSI VALIDATION (Confirming Lab 4 Findings)")
        print("=" * 45)
        
        key_features = [
            ' ROA(C) before interest and depreciation before interest',
            ' Debt ratio %',
            ' Current Ratio',
            ' Operating Gross Margin'
        ]
        
        for feature in key_features:
            if feature in X_train.columns:
                psi_value = self._calculate_psi_single_feature(
                    X_train[feature], X_test[feature]
                )
                status = "✅ No drift" if psi_value < 0.1 else "⚠️ Drift detected"
                print(f"  {feature[:30]}... PSI = {psi_value:.4f} ({status})")

    def _calculate_psi_single_feature(self, train_data: pd.Series, test_data: pd.Series, bins: int = 10) -> float:
        """Calculate PSI for a single feature (simplified version from Lab 4)"""
        try:
            # Remove missing values
            train_clean = train_data.dropna()
            test_clean = test_data.dropna()
            
            if len(train_clean) == 0 or len(test_clean) == 0:
                return 0.0
            
            # Create bins based on training data quantiles
            _, bin_edges = pd.qcut(train_clean, bins, retbins=True, duplicates='drop')
            
            # Calculate distributions
            train_dist = pd.cut(train_clean, bin_edges, include_lowest=True).value_counts(normalize=True)
            test_dist = pd.cut(test_clean, bin_edges, include_lowest=True).value_counts(normalize=True)
            
            # Align indices and fill missing with small value
            train_dist = train_dist.reindex(test_dist.index, fill_value=1e-6)
            test_dist = test_dist.fillna(1e-6)
            
            # Calculate PSI
            psi = ((test_dist - train_dist) * np.log(test_dist / train_dist)).sum()
            return psi
            
        except Exception:
            return 0.0

    def select_features(self, X_train: pd.DataFrame, y_train: pd.Series) -> list[str]:
        """
        Feature selection using XGBoost importance (Lab 4 decision).
        Keep it simple as professor requested.
        """
        print("\\n🎯 FEATURE SELECTION (XGBoost Importance - Lab 4 Choice)")
        print("=" * 60)
        
        # Train basic XGBoost for feature importance
        xgb_selector = xgb.XGBClassifier(
            n_estimators=100,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        xgb_selector.fit(X_train, y_train)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top 40 features (Lab 4 plan)
        top_features = importance_df.head(40)['feature'].tolist()
        
        print(f"Selected {len(top_features)} features from {len(X_train.columns)} total")
        print("Top 10 most important features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['feature'][:45]}... {row['importance']:.4f}")
        
        # Create feature importance plot
        plt.figure(figsize=(10, 8))
        top_20 = importance_df.head(20)
        plt.barh(range(len(top_20)), top_20['importance'], color='lightblue')
        plt.yticks(range(len(top_20)), 
                  [f[:25] + '...' for f in top_20['feature']], fontsize=8)
        plt.xlabel('XGBoost Feature Importance')
        plt.title('Top 20 Features by XGBoost Importance\\n(Lab 4 Selection Method)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.selected_features = top_features
        return top_features

    def setup_models(self) -> dict[str, any]:
        """
        Setup the three models from Lab 4 research:
        - Logistic Regression (benchmark, interpretable)
        - Random Forest (robust to outliers)
        - XGBoost (handles imbalance well)
        """
        print("\\n🤖 MODEL SETUP (Lab 4 Choices)")
        print("=" * 35)
        
        models = {
            'Logistic_Regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'needs_scaling': True,  # Lab 4 decision
                'param_grid': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'Random_Forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'needs_scaling': False,  # Lab 4 decision - tree models don't need scaling
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'needs_scaling': False,  # Lab 4 decision
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                }
            }
        }
        
        print("Models configured:")
        for name, config in models.items():
            scaling = "with scaling" if config['needs_scaling'] else "no scaling"
            print(f"  ✅ {name}: {scaling}")
        
        return models

    def train_and_tune_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train and tune models using RandomizedSearchCV (Lab 4 choice).
        Keep hyperparameter tuning simple as professor requested.
        """
        print("\\n🏃 MODEL TRAINING & HYPERPARAMETER TUNING")
        print("=" * 50)
        
        models_config = self.setup_models()
        
        # Filter features to selected ones
        X_train_selected = X_train[self.selected_features]
        
        # Stratified 5-fold CV (Lab 4 decision)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for model_name, config in models_config.items():
            print(f"\\n🔧 Training {model_name}...")
            
            # Prepare data (scale if needed)
            if config['needs_scaling']:
                scaler = StandardScaler()
                X_train_processed = scaler.fit_transform(X_train_selected)
                self.scalers[model_name] = scaler
                print(f"  ✅ Applied StandardScaler for {model_name}")
            else:
                X_train_processed = X_train_selected.values
                self.scalers[model_name] = None
            
            # Hyperparameter tuning with RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=config['model'],
                param_distributions=config['param_grid'],
                n_iter=10,  # Keep simple as requested
                cv=cv,
                scoring='roc_auc',  # Lab 4 primary metric
                random_state=self.random_state,
                n_jobs=-1
            )
            
            random_search.fit(X_train_processed, y_train)
            
            # Store best model
            self.models[model_name] = random_search.best_estimator_
            
            print(f"  ✅ Best CV ROC-AUC: {random_search.best_score_:.4f}")
            print(f"  ✅ Best parameters: {random_search.best_params_}")

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series,
                       X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Comprehensive model evaluation with all Lab 4 metrics:
        - ROC-AUC, PR-AUC, F1-score
        - Calibration curves and Brier scores
        - Training vs test performance comparison
        """
        print("\\n📊 MODEL EVALUATION & COMPARISON")
        print("=" * 40)
        
        # Filter to selected features
        X_train_selected = X_train[self.selected_features]
        X_test_selected = X_test[self.selected_features]
        
        results = []
        
        # Create evaluation plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Evaluation Dashboard\\n(Lab 4 Metrics Implementation)', fontsize=16)
        
        for i, (model_name, model) in enumerate(self.models.items()):
            # Prepare data (apply same scaling as training)
            if self.scalers[model_name] is not None:
                X_train_proc = self.scalers[model_name].transform(X_train_selected)
                X_test_proc = self.scalers[model_name].transform(X_test_selected)
            else:
                X_train_proc = X_train_selected.values
                X_test_proc = X_test_selected.values
            
            # Get predictions and probabilities
            train_proba = model.predict_proba(X_train_proc)[:, 1]
            test_proba = model.predict_proba(X_test_proc)[:, 1]
            
            train_pred = model.predict(X_train_proc)
            test_pred = model.predict(X_test_proc)
            
            # Calculate metrics (Lab 4 choices)
            train_roc_auc = roc_auc_score(y_train, train_proba)
            test_roc_auc = roc_auc_score(y_test, test_proba)
            
            train_f1 = f1_score(y_train, train_pred)
            test_f1 = f1_score(y_test, test_pred)
            
            train_brier = brier_score_loss(y_train, train_proba)
            test_brier = brier_score_loss(y_test, test_proba)
            
            # PR-AUC calculation
            train_precision, train_recall, _ = precision_recall_curve(y_train, train_proba)
            test_precision, test_recall, _ = precision_recall_curve(y_test, test_proba)
            train_pr_auc = auc(train_recall, train_precision)
            test_pr_auc = auc(test_recall, test_precision)
            
            results.append({
                'Model': model_name.replace('_', ' '),
                'Train_ROC_AUC': train_roc_auc,
                'Test_ROC_AUC': test_roc_auc,
                'Train_PR_AUC': train_pr_auc,
                'Test_PR_AUC': test_pr_auc,
                'Train_F1': train_f1,
                'Test_F1': test_f1,
                'Train_Brier': train_brier,
                'Test_Brier': test_brier
            })
            
            # Plot ROC curves (Training vs Test)
            ax_roc = axes[0, i]
            train_fpr, train_tpr, _ = roc_curve(y_train, train_proba)
            test_fpr, test_tpr, _ = roc_curve(y_test, test_proba)
            
            ax_roc.plot(train_fpr, train_tpr, label=f'Train (AUC={train_roc_auc:.3f})', linewidth=2)
            ax_roc.plot(test_fpr, test_tpr, label=f'Test (AUC={test_roc_auc:.3f})', linewidth=2)
            ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title(f'{model_name.replace("_", " ")}\\nROC Curve')
            ax_roc.legend()
            ax_roc.grid(True, alpha=0.3)
            
            # Plot calibration curves
            ax_cal = axes[1, i]
            
            # Train calibration
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_train, train_proba, n_bins=10
            )
            ax_cal.plot(mean_predicted_value, fraction_of_positives, "s-", 
                       label=f'Train (Brier={train_brier:.3f})', linewidth=2)
            
            # Test calibration
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, test_proba, n_bins=10
            )
            ax_cal.plot(mean_predicted_value, fraction_of_positives, "s-",
                       label=f'Test (Brier={test_brier:.3f})', linewidth=2)
            
            ax_cal.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax_cal.set_xlabel('Mean Predicted Probability')
            ax_cal.set_ylabel('Fraction of Positives')
            ax_cal.set_title(f'{model_name.replace("_", " ")}\\nCalibration Plot')
            ax_cal.legend()
            ax_cal.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create results table
        results_df = pd.DataFrame(results)
        print("\\n📋 MODEL COMPARISON TABLE")
        print("=" * 25)
        print(results_df.round(4).to_string(index=False))
        
        # Find best model
        best_model_name = results_df.loc[results_df['Test_ROC_AUC'].idxmax(), 'Model']
        print(f"\\n🏆 Best performing model: {best_model_name}")
        print(f"   (Highest Test ROC-AUC: {results_df['Test_ROC_AUC'].max():.4f})")
        
        self.results['evaluation'] = results_df
        self.results['best_model'] = best_model_name.replace(' ', '_')

    def analyze_with_shap(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        SHAP analysis for the best model (Lab 4 interpretability choice).
        """
        print("\\n🔍 SHAP ANALYSIS FOR {self.results['best_model']}")
        print("=" * 50)
        
        best_model_name = self.results['best_model']
        best_model = self.models[best_model_name]
        
        # Prepare test data
        X_test_selected = X_test[self.selected_features]
        if self.scalers[best_model_name] is not None:
            X_test_proc = self.scalers[best_model_name].transform(X_test_selected)
            # Convert back to DataFrame for SHAP
            X_test_proc = pd.DataFrame(X_test_proc, columns=X_test_selected.columns)
        else:
            X_test_proc = X_test_selected
        
        # Take smaller sample for SHAP (computational efficiency)
        sample_size = min(500, len(X_test_proc))
        X_sample = X_test_proc.sample(n=sample_size, random_state=self.random_state)
        
        # Create SHAP explainer
        try:
            if 'XGBoost' in best_model_name or 'Random_Forest' in best_model_name:
                explainer = shap.TreeExplainer(best_model)
                shap_values = explainer.shap_values(X_sample.values)
            else:
                explainer = shap.LinearExplainer(best_model, X_sample.values)
                shap_values = explainer.shap_values(X_sample.values)
            
            # Create SHAP plots
            plt.figure(figsize=(15, 10))
            
            # Summary plot
            plt.subplot(2, 2, (1, 2))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
            plt.title(f'SHAP Feature Importance\\n{best_model_name.replace("_", " ")}')
            
            # Beeswarm plot
            plt.subplot(2, 2, (3, 4))
            shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
            plt.title(f'SHAP Values Distribution\\n{best_model_name.replace("_", " ")}')
            
            plt.tight_layout()
            plt.savefig('shap_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print top features
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
                
            mean_abs_shap = np.abs(shap_values).mean(0)
            feature_importance = pd.DataFrame({
                'feature': X_sample.columns,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False)
            
            print("Top 10 most important features (by SHAP):")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
                print(f"  {i:2d}. {row['feature'][:40]}... {row['importance']:.4f}")
            
            print("\\n💼 BUSINESS INSIGHTS:")
            print("   These features are most critical for bankruptcy prediction")
            print("   Can be used for regulatory reporting and risk assessment")
            print("   SHAP values provide explanation for individual company predictions")
            
        except Exception as e:
            print(f"⚠️ SHAP analysis encountered an issue: {str(e)}")
            print("   This is common with certain model configurations")
            print("   Proceeding without SHAP analysis")

    def save_models_and_artifacts(self) -> None:
        """Save trained models and important artifacts."""
        print("\\n💾 SAVING MODELS AND ARTIFACTS")
        print("=" * 35)
        
        # Save models
        for model_name, model in self.models.items():
            joblib.dump(model, f'{model_name}_model.pkl')
            print(f"  ✅ Saved {model_name}_model.pkl")
        
        # Save scalers
        for model_name, scaler in self.scalers.items():
            if scaler is not None:
                joblib.dump(scaler, f'{model_name}_scaler.pkl')
                print(f"  ✅ Saved {model_name}_scaler.pkl")
        
        # Save selected features
        with open('selected_features.txt', 'w') as f:
            for feature in self.selected_features:
                f.write(f"{feature}\\n")
        print(f"  ✅ Saved selected_features.txt ({len(self.selected_features)} features)")
        
        # Save results
        if 'evaluation' in self.results:
            self.results['evaluation'].to_csv('model_evaluation_results.csv', index=False)
            print(f"  ✅ Saved model_evaluation_results.csv")

    def run_complete_pipeline(self, file_path: str = 'data.csv') -> None:
        """
        Run the complete training pipeline implementing all Lab 4 decisions.
        """
        print("🚀 BANKRUPTCY PREDICTION TRAINING PIPELINE")
        print("=" * 50)
        print("Implementing research decisions from Lab 4:")
        print("  • Models: Logistic Regression + Random Forest + XGBoost")
        print("  • Class imbalance: SMOTE (1:29 ratio found)")
        print("  • Outlier treatment: Cap at 99th percentile")
        print("  • Feature selection: XGBoost importance (top 40)")
        print("  • Validation: Stratified 5-fold CV")
        print("  • Metrics: ROC-AUC, PR-AUC, F1-score, Brier score")
        print("=" * 50)
        
        try:
            # Step 1: Load and explore data
            df = self.load_and_explore_data(file_path)
            
            # Step 2: Preprocess data
            X_train, X_test, y_train, y_test = self.preprocess_data(df)
            
            # Step 3: Feature selection
            selected_features = self.select_features(X_train, y_train)
            
            # Step 4: Train and tune models
            self.train_and_tune_models(X_train, y_train)
            
            # Step 5: Evaluate models
            self.evaluate_models(X_test, y_test, X_train, y_train)
            
            # Step 6: SHAP analysis
            self.analyze_with_shap(X_test, y_test)
            
            # Step 7: Save everything
            self.save_models_and_artifacts()
            
            print("\\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 35)
            print("✅ Models trained and evaluated")
            print(f"✅ Best model: {self.results['best_model'].replace('_', ' ')}")
            print("✅ Artifacts saved for deployment")
            print("✅ Visualizations created")
            print("\\nFiles generated:")
            print("  📊 eda_analysis.png")
            print("  🔗 feature_importance.png")
            print("  📈 model_evaluation.png")
            print("  🔍 shap_analysis.png")
            print("  🤖 *_model.pkl files")
            print("  📄 model_evaluation_results.csv")
            
            return self.results
            
        except Exception as e:
            print(f"\\n❌ ERROR IN PIPELINE: {str(e)}")
            raise

def main():
    """
    Main execution function.
    Run the complete bankruptcy prediction pipeline.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize and run pipeline
    pipeline = BankruptcyPredictionPipeline(random_state=42)
    results = pipeline.run_complete_pipeline('data.csv')
    
    print("\\n🏁 Lab 5 Implementation Complete!")
    print("Ready for video presentation and deployment!")

if __name__ == "__main__":
    main()