"""
Machine Learning Models for Ballon d'Or Prediction
Implements and evaluates multiple algorithms as specified in the project requirements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.model_selection import cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
import json
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BallonDorModelTrainer:
    """Trains and evaluates multiple ML models for Ballon d'Or prediction"""

    def __init__(self, model_dir: str = "../models", reports_dir: str = "../reports"):
        self.model_dir = model_dir
        self.reports_dir = reports_dir
        self.models = {}
        self.model_performance = {}
        self.best_model = None
        self.best_model_name = None

        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all models to be tested
        Returns dictionary of model names and instances
        """
        models = {
            # Tree-based models
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1
            ),
            'CatBoost': cb.CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_seed=42,
                verbose=False
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),

            # Linear models
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'BayesianRidge': BayesianRidge(random_state=42),

            # Other models
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'KNeighbors': KNeighborsRegressor(n_neighbors=5),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)  # For classification approach
        }

        return models

    def train_model(self, model_name: str, model: Any, X_train: pd.DataFrame,
                   y_train: pd.Series, X_val: pd.DataFrame = None,
                   y_val: pd.Series = None) -> Dict[str, float]:
        """
        Train a single model and evaluate its performance
        """
        logger.info(f"Training {model_name}...")

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val) if X_val is not None else None

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)

        metrics = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2
        }

        if X_val is not None and y_val is not None:
            val_mae = mean_absolute_error(y_val, y_pred_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            val_r2 = r2_score(y_val, y_pred_val)

            metrics.update({
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2
            })

        # Store model and performance
        self.models[model_name] = model
        self.model_performance[model_name] = metrics

        logger.info(f"{model_name} - Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}, Train R²: {train_r2:.4f}")
        if X_val is not None:
            logger.info(f"{model_name} - Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}, Val R²: {val_r2:.4f}")

        return metrics

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame = None, y_val: pd.Series = None,
                        use_cross_validation: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Train all models and return performance metrics
        """
        logger.info("Starting training of all models...")

        models = self.initialize_models()

        if use_cross_validation and X_val is None:
            # Use time series cross-validation for temporal data
            tscv = TimeSeriesSplit(n_splits=5)
            cv_results = {}

            for name, model in models.items():
                logger.info(f"Cross-validating {name}...")
                cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
                cv_mae = -cv_scores.mean()
                cv_std = cv_scores.std()

                # Train on full data for final model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_train)
                train_mae = mean_absolute_error(y_train, y_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
                train_r2 = r2_score(y_train, y_pred)

                self.models[name] = model
                self.model_performance[name] = {
                    'train_mae': train_mae,
                    'train_rmse': train_rmse,
                    'train_r2': train_r2,
                    'cv_mae': cv_mae,
                    'cv_mae_std': cv_std
                }

                logger.info(f"{name} - CV MAE: {cv_mae:.4f} (±{cv_std:.4f}), Train MAE: {train_mae:.4f}")

        else:
            # Train/validation split approach
            for name, model in models.items():
                self.train_model(name, model, X_train, y_train, X_val, y_val)

        return self.model_performance

    def create_ensemble_model(self, model_names: List[str] = None,
                            voting_type: str = 'soft') -> VotingRegressor:
        """
        Create an ensemble model from top-performing models
        """
        if model_names is None:
            # Select top 3 models based on validation performance
            if self.model_performance:
                sorted_models = sorted(
                    self.model_performance.items(),
                    key=lambda x: x[1].get('val_mae', x[1].get('train_mae', float('inf')))
                )
                model_names = [name for name, _ in sorted_models[:3]]
            else:
                # Default to top performers
                model_names = ['RandomForest', 'XGBoost', 'LightGBM']

        estimators = [(name, self.models[name]) for name in model_names if name in self.models]
        ensemble = VotingRegressor(estimators=estimators, weights=None, voting=voting_type)

        logger.info(f"Created ensemble with models: {[name for name, _ in estimators]}")
        return ensemble

    def evaluate_model(self, model: Any, X_test: pd.DataFrame,
                      y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on test set
        """
        y_pred = model.predict(X_test)

        metrics = {
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_r2': r2_score(y_test, y_pred)
        }

        # For ranking metrics, we can calculate top-k accuracy
        # Convert to ranking problem: top 1, top 3 accuracy
        if len(y_test) > 1:
            # Get indices sorted by actual and predicted values
            actual_rank = np.argsort(y_test.values)[::-1]  # Descending order
            pred_rank = np.argsort(y_pred)[::-1]  # Descending order

            # Top-1 accuracy
            top1_actual = set(actual_rank[:1])
            top1_pred = set(pred_rank[:1])
            top1_accuracy = len(top1_actual & top1_pred) / len(top1_actual) if top1_actual else 0

            # Top-3 accuracy
            top3_actual = set(actual_rank[:3])
            top3_pred = set(pred_rank[:3])
            top3_accuracy = len(top3_actual & top3_pred) / len(top3_actual) if top3_actual else 0

            metrics.update({
                'top1_accuracy': top1_accuracy,
                'top3_accuracy': top3_accuracy
            })

        return metrics

    def save_model(self, model: Any, model_name: str):
        """
        Save trained model to disk
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Model {model_name} saved to {model_path}")

    def save_all_models(self):
        """
        Save all trained models
        """
        for name, model in self.models.items():
            self.save_model(model, name)

        # Save model performance report
        report_path = os.path.join(self.reports_dir, "model_performance.json")
        with open(report_path, 'w') as f:
            json.dump(self.model_performance, f, indent=2, default=str)
        logger.info(f"Model performance report saved to {report_path}")

    def load_model(self, model_name: str) -> Any:
        """
        Load trained model from disk
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Model {model_name} loaded from {model_path}")
            return model
        else:
            logger.error(f"Model {model_name} not found at {model_path}")
            return None

    def get_model_rankings(self) -> List[Tuple[str, Dict[str, float]]]:
        """
        Return models ranked by performance (lower MAE is better)
        """
        if not self.model_performance:
            return []

        # Use validation MAE if available, otherwise training MAE
        def get_mae(metrics):
            return metrics.get('val_mae', metrics.get('train_mae', float('inf')))

        sorted_models = sorted(
            self.model_performance.items(),
            key=lambda x: get_mae(x[1])
        )
        return sorted_models

    def get_best_model(self) -> Tuple[str, Any]:
        """
        Return the best performing model and its name
        """
        rankings = self.get_model_rankings()
        if rankings:
            best_name, _ = rankings[0]
            return best_name, self.models[best_name]
        return None, None

    def predict_ballon_dor(self, model: Any, X: pd.DataFrame,
                          player_names: List[str] = None) -> pd.DataFrame:
        """
        Generate Ballon d'Or predictions using a trained model
        """
        predictions = model.predict(X)

        results_df = pd.DataFrame({
            'player_name': player_names if player_names else [f'Player_{i}' for i in range(len(predictions))],
            'predicted_points': predictions,
            'rank': np.argsort(predictions)[::-1].argsort() + 1  # Rank 1 = highest points
        })

        # Sort by predicted points descending
        results_df = results_df.sort_values('predicted_points', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)

        return results_df

    def generate_model_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive model report for README
        """
        report = {
            'models_tested': list(self.models.keys()),
            'model_performance': self.model_performance,
            'best_model': self.best_model_name,
            'best_model_performance': self.model_performance.get(self.best_model_name, {}),
            'timestamp': datetime.now().isoformat()
        }

        # Save report
        report_path = os.path.join(self.reports_dir, "model_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Model report saved to {report_path}")

        return report


def create_model_registry() -> Dict[str, Dict[str, str]]:
    """
    Create a registry of all models used in the project for documentation
    """
    registry = {
        'RandomForest': {
            'type': 'Ensemble Tree-based',
            'purpose': 'Robust baseline model, handles non-linear relationships well',
            'key_hyperparameters': 'n_estimators=100, max_depth=10, min_samples_split=5',
            'strengths': 'Handles overfitting well, interpretable, good default performance',
            'weaknesses': 'Can be slow on large datasets, less flexible than boosting methods'
        },
        'XGBoost': {
            'type': 'Gradient Boosting',
            'purpose': 'High-performance gradient boosting with regularization',
            'key_hyperparameters': 'n_estimators=100, max_depth=6, learning_rate=0.1',
            'strengths': 'Excellent performance, handles missing values, feature importance',
            'weaknesses': 'Can overfit if not tuned properly, sensitive to hyperparameters'
        },
        'LightGBM': {
            'type': 'Gradient Boosting',
            'purpose': 'Fast gradient boosting with leaf-wise tree growth',
            'key_hyperparameters': 'n_estimators=100, max_depth=6, learning_rate=0.1',
            'strengths': 'Very fast, handles large datasets well, good accuracy',
            'weaknesses': 'Can overfit on small datasets, less interpretable'
        },
        'CatBoost': {
            'type': 'Gradient Boosting',
            'purpose': 'Handles categorical features well, robust to overfitting',
            'key_hyperparameters': 'iterations=100, depth=6, learning_rate=0.1',
            'strengths': 'Handles categorical features natively, robust, good default params',
            'weaknesses': 'Slower than LightGBM, longer training time'
        },
        'GradientBoosting': {
            'type': 'Gradient Boosting',
            'purpose': 'Classic gradient boosting algorithm',
            'key_hyperparameters': 'n_estimators=100, max_depth=6, learning_rate=0.1',
            'strengths': 'Well-understood, good baseline boosting method',
            'weaknesses': 'Generally outperformed by XGBoost/LightGBM/CatBoost'
        },
        'Ridge': {
            'type': 'Linear Regularized',
            'purpose': 'Linear model with L2 regularization to prevent overfitting',
            'key_hyperparameters': 'alpha=1.0',
            'strengths': 'Fast, interpretable, works well with many features',
            'weaknesses': 'Assumes linear relationships, may underfit complex patterns'
        },
        'ElasticNet': {
            'type': 'Linear Regularized',
            'purpose': 'Linear model with L1 and L2 regularization',
            'key_hyperparameters': 'alpha=0.1, l1_ratio=0.5',
            'strengths': 'Feature selection via L1, handles multicollinearity',
            'weaknesses': 'Still assumes linear relationships'
        },
        'BayesianRidge': {
            'type': 'Linear Probabilistic',
            'purpose': 'Bayesian ridge regression with automatic alpha tuning',
            'key_hyperparameters': 'Non-informative priors',
            'strengths': 'Automatic regularization, uncertainty estimates',
            'weaknesses': 'Assumes linear relationships, Bayesian computation overhead'
        },
        'SVR': {
            'type': 'Kernel-based',
            'purpose': 'Support Vector Regression with RBF kernel',
            'key_hyperparameters': 'kernel=rbf, C=1.0, gamma=scale',
            'strengths': 'Effective in high-dimensional spaces, flexible via kernel trick',
            'weaknesses': 'Slow on large datasets, sensitive to feature scaling'
        },
        'KNeighbors': {
            'type': 'Instance-based',
            'purpose': 'Simple baseline based on similarity to training instances',
            'key_hyperparameters': 'n_neighbors=5',
            'strengths': 'Simple to understand, non-parametric',
            'weaknesses': 'Slow prediction, sensitive to scale, suffers from curse of dimensionality'
        },
        'VotingEnsemble': {
            'type': 'Ensemble Meta-model',
            'purpose': 'Combines predictions from multiple models for improved robustness',
            'key_hyperparameters': 'VotingRegressor with top 3 performers',
            'strengths': 'Reduces variance, often better than individual models',
            'weaknesses': 'Increased complexity, longer prediction time'
        }
    }

    return registry


if __name__ == "__main__":
    # Example usage with synthetic data
    logger.info("Running example model training...")

    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 100
    n_features = 15

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Create target with some relationship to features
    y = pd.Series(
        X['feature_0'] * 2 + X['feature_1'] * 1.5 + X['feature_2'] * -1 +
        np.random.randn(n_samples) * 0.5 + 10
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize trainer
    trainer = BallonDorModelTrainer()

    # Train all models
    performance = trainer.train_all_models(X_train, y_train, X_test, y_test, use_cross_validation=False)

    # Get best model
    best_name, best_model = trainer.get_best_model()
    logger.info(f"Best model: {best_name}")

    # Evaluate on test set
    if best_model:
        test_metrics = trainer.evaluate_model(best_model, X_test, y_test)
        logger.info(f"Test performance: {test_metrics}")

    # Save models
    trainer.save_all_models()

    # Generate model registry
    registry = create_model_registry()
    registry_path = os.path.join(trainer.reports_dir, "model_registry.json")
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    logger.info(f"Model registry saved to {registry_path}")

    print("\nModel training example completed!")
    print(f"Trained {len(trainer.models)} models")
    print(f"Best model: {best_name}")