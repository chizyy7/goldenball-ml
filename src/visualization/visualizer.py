"""
Visualization Module for Ballon d'Or Prediction
Generates all required charts and graphs for the project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from typing import Dict, List, Tuple, Any
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BallonDorVisualizer:
    """Creates visualizations for Ballon d'Or prediction project"""

    def __init__(self, assets_dir: str = "../assets"):
        self.assets_dir = assets_dir
        os.makedirs(self.assets_dir, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_feature_importance(self, feature_names: List[str],
                              importances: np.ndarray,
                              title: str = "Feature Importance",
                              top_n: int = 15) -> str:
        """
        Create feature importance bar chart
        """
        logger.info("Creating feature importance bar chart...")

        # Get top N features
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        # Create plot
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_features)), top_importances)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.gca().invert_yaxis()  # Highest importance at top

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + max(top_importances) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{top_importances[i]:.3f}', ha='left', va='center')

        plt.tight_layout()

        # Save plot
        filename = "feature_importance.png"
        filepath = os.path.join(self.assets_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Feature importance chart saved to {filepath}")
        return filepath

    def plot_radar_chart(self, candidates_data: Dict[str, List[float]],
                        feature_names: List[str],
                        title: str = "Candidate Comparison Radar Chart") -> str:
        """
        Create radar chart comparing top candidates
        """
        logger.info("Creating radar chart for candidate comparison...")

        # Number of variables
        N = len(feature_names)

        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Plot each candidate
        colors = plt.cm.Set1(np.linspace(0, 1, len(candidates_data)))
        for i, (candidate, values) in enumerate(candidates_data.items()):
            # Normalize values for radar chart (0-1 scale)
            values_norm = np.array(values)
            if np.max(values_norm) - np.min(values_norm) > 0:
                values_norm = (values_norm - np.min(values_norm)) / (np.max(values_norm) - np.min(values_norm))
            else:
                values_norm = np.zeros_like(values_norm)

            values_norm = np.concatenate((values_norm, [values_norm[0]]))  # Complete the circle

            ax.plot(angles, values_norm, 'o-', linewidth=2, label=candidate, color=colors[i])
            ax.fill(angles, values_norm, alpha=0.25, color=colors[i])

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names)
        ax.set_ylim(0, 1)
        ax.set_title(title, size=16, y=1.08)
        ax.grid(True)

        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()

        # Save plot
        filename = "radar_chart.png"
        filepath = os.path.join(self.assets_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Radar chart saved to {filepath}")
        return filepath

    def plot_predicted_vs_actual(self, actual_points: np.ndarray,
                               predicted_points: np.ndarray,
                               player_names: List[str] = None,
                               title: str = "Predicted vs Actual Points") -> str:
        """
        Create scatter plot of predicted vs actual points
        """
        logger.info("Creating predicted vs actual scatter plot...")

        plt.figure(figsize=(10, 8))

        # Scatter plot
        plt.scatter(actual_points, predicted_points, alpha=0.7, s=60)

        # Perfect prediction line
        max_val = max(np.max(actual_points), np.max(predicted_points))
        min_val = min(np.min(actual_points), np.min(predicted_points))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        # Add labels for top performers if player names provided
        if player_names is not None and len(player_names) == len(actual_points):
            # Label top 5 actual and predicted
            actual_top_idx = np.argsort(actual_points)[::-1][:5]
            pred_top_idx = np.argsort(predicted_points)[::-1][:5]

            for idx in actual_top_idx:
                plt.annotate(player_names[idx],
                           (actual_points[idx], predicted_points[idx]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)

        plt.xlabel('Actual Points')
        plt.ylabel('Predicted Points')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add R² and MAE
        mae = mean_absolute_error(actual_points, predicted_points)
        rmse = np.sqrt(mean_squared_error(actual_points, predicted_points))
        r2 = r2_score(actual_points, predicted_points)

        plt.text(0.05, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        # Save plot
        filename = "predicted_vs_actual.png"
        filepath = os.path.join(self.assets_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Predicted vs actual chart saved to {filepath}")
        return filepath

    def plot_waterfall_chart(self, feature_names: List[str],
                           feature_values: np.ndarray,
                           base_value: float,
                           prediction: float,
                           title: str = "Feature Contribution Waterfall Chart") -> str:
        """
        Create waterfall chart showing feature contributions to prediction
        """
        logger.info("Creating feature contribution waterfall chart...")

        # Calculate the contribution of the last element (to make sum equal to prediction)
        # The sum of base_value + all contributions should equal prediction
        # So the last contribution = prediction - base_value - sum(other contributions)
        if len(feature_values) > 0:
            contributions = np.append(feature_values[:-1], prediction - base_value - np.sum(feature_values[:-1]))
        else:
            contributions = np.array([prediction - base_value])

        cumulative = np.cumsum(np.insert(contributions, 0, base_value))

        # Create plot
        plt.figure(figsize=(12, 8))

        # Bars for each feature
        x_pos = np.arange(len(feature_names) + 2)  # +2 for base and prediction
        bar_values = np.insert(contributions, 0, [base_value, prediction - np.sum(contributions) - base_value])
        bar_values = bar_values[:-1]  # Remove last element to match x_pos

        # Ensure bar_values matches x_pos length
        if len(bar_values) != len(x_pos):
            # Truncate or pad to match
            if len(bar_values) > len(x_pos):
                bar_values = bar_values[:len(x_pos)]
            else:
                # Pad with zeros
                bar_values = np.pad(bar_values, (0, len(x_pos) - len(bar_values)), 'constant')

        colors = []
        for val in bar_values:
            if val >= 0:
                colors.append('green')
            else:
                colors.append('red')

        # Create bars
        bars = plt.bar(x_pos, bar_values, color=colors, alpha=0.7, edgecolor='black')

        # Connect with lines
        line_x = np.repeat(x_pos, 2)
        line_y = np.zeros(len(line_x))
        # Only set values where we have data
        min_len = min(len(line_y)//2, len(cumulative)-1)
        if min_len > 0:
            line_y[::2][:min_len] = cumulative[1:min_len+1]
            line_y[1::2][:min_len] = cumulative[1:min_len+1]
        plt.plot(line_x, line_y, 'k-', linewidth=1.5)

        # Customize plot
        labels = ['Base'] + list(feature_names) + ['Prediction']
        plt.xticks(x_pos, labels, rotation=45, ha='right')
        plt.ylabel('Contribution to Points')
        plt.title(title)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, bar_values)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2,
                    height + (0.01 * max(np.abs(bar_values))) if height >= 0
                    else height - (0.03 * max(np.abs(bar_values))),
                    f'{val:.1f}', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8)

        plt.tight_layout()

        # Save plot
        filename = "waterfall_chart.png"
        filepath = os.path.join(self.assets_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Waterfall chart saved to {filepath}")
        return filepath

    def plot_confidence_intervals(self, predictions_df: pd.DataFrame,
                                confidence_lower: np.ndarray,
                                confidence_upper: np.ndarray,
                                title: str = "Top 10 Predictions with Confidence Intervals") -> str:
        """
        Create confidence interval plot for top 10 predictions
        """
        logger.info("Creating confidence interval plot...")

        # Sort by predicted points descending and take top 10
        top_10 = predictions_df.head(10).copy()

        plt.figure(figsize=(12, 8))

        # Create error bars
        x_pos = range(len(top_10))
        predicted_points = top_10['predicted_points'].values
        lower_err = predicted_points - confidence_lower[:len(top_10)]
        upper_err = confidence_upper[:len(top_10)] - predicted_points
        errors = [lower_err, upper_err]

        # Create bar plot with error bars
        bars = plt.bar(x_pos, predicted_points, yerr=errors, capsize=5,
                      alpha=0.7, color='skyblue', edgecolor='navy')

        # Customize plot
        plt.xlabel('Players (Ranked by Predicted Points)')
        plt.ylabel('Predicted Points')
        plt.title(title)
        plt.xticks(x_pos, top_10['player_name'], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, pred) in enumerate(zip(bars, predicted_points)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(upper_err)*0.01,
                    f'{pred:.1f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Save plot
        filename = "confidence_intervals.png"
        filepath = os.path.join(self.assets_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Confidence interval chart saved to {filepath}")
        return filepath

    def plot_correlation_heatmap(self, df: pd.DataFrame,
                               title: str = "Feature Correlation Heatmap") -> str:
        """
        Create correlation heatmap of key features
        """
        logger.info("Creating correlation heatmap...")

        # Select numeric columns for correlation
        numeric_df = df.select_dtypes(include=[np.number])

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()

        # Create plot
        plt.figure(figsize=(12, 10))

        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})

        plt.title(title, fontsize=16, pad=20)
        plt.tight_layout()

        # Save plot
        filename = "correlation_heatmap.png"
        filepath = os.path.join(self.assets_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Correlation heatmap saved to {filepath}")
        return filepath

    def plot_model_comparison(self, model_performance: Dict[str, Dict[str, float]],
                            metric: str = 'val_mae',
                            title: str = "Model Performance Comparison") -> str:
        """
        Create model comparison chart
        """
        logger.info("Creating model comparison chart...")

        # Extract model names and performance metric
        models = list(model_performance.keys())
        if models and metric in model_performance[models[0]]:
            values = [model_performance[model].get(model_performance[model].get('val_mae', model_performance[model].get('train_mae', 0)), 0)
                     for model in models]
            # Fix the above line - let's do it properly
            values = []
            for model in models:
                perf = model_performance[model]
                val = perf.get('val_mae', perf.get('train_mae', 0))
                values.append(val)
        else:
            # Fallback to first available metric
            values = []
            for model in models:
                perf = model_performance[model]
                # Get first numeric value that's not a std
                numeric_vals = [v for k, v in perf.items() if isinstance(v, (int, float)) and 'std' not in k.lower()]
                values.append(numeric_vals[0] if numeric_vals else 0)

        # Sort by performance (lower is better for error metrics)
        sorted_indices = np.argsort(values)
        sorted_models = [models[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]

        # Create plot
        plt.figure(figsize=(12, 8))

        bars = plt.bar(range(len(sorted_models)), sorted_values, alpha=0.7, color='steelblue')
        plt.xlabel('Models')
        plt.ylabel(f'{metric.upper()} (Lower is Better)')
        plt.title(title)
        plt.xticks(range(len(sorted_models)), sorted_models, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, sorted_values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sorted_values)*0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Save plot
        filename = "model_comparison.png"
        filepath = os.path.join(self.assets_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Model comparison chart saved to {filepath}")
        return filepath

    def create_pipeline_diagram(self) -> str:
        """
        Create Mermaid pipeline diagram for README
        """
        logger.info("Creating Mermaid pipeline diagram...")

        mermaid_code = '''
        flowchart TD
            A[Data Acquisition] --> B[Historical Ballon d'Or Data]
            A --> C[Player Statistics (FBref, etc.)]
            A --> D[Team Achievements Data]
            A --> E[Contextual/Narrative Data]

            B --> F[Data Preprocessing]
            C --> F
            D --> F
            E --> F

            F --> G[Feature Engineering]
            G --> H[Individual Performance Features<br/>(Goals, Assists, xG, etc.)]
            G --> I[Collective Achievement Features<br/>(UCL, League Titles, etc.)]
            G --> J[Contextual Features<br/>(Media, Social, Narrative)]

            H --> K[Feature Combination]
            I --> K
            J --> K

            K --> L[Feature Scaling & Selection]
            L --> M[Model Training]

            M --> N[Random Forest]
            M --> O[XGBoost]
            M --> P[LightGBM]
            M --> Q[CatBoost]
            M --> R[Gradient Boosting]
            M --> S[Ridge Regression]
            M --> T[ElasticNet]
            M --> U[Bayesian Ridge]
            M --> V[SVR]
            M --> W[K-Nearest Neighbors]

            N --> X[Model Evaluation]
            O --> X
            P --> X
            Q --> X
            R --> X
            S --> X
            T --> X
            U --> X
            V --> X
            W --> X

            X --> Y[Best Model Selection]
            Y --> Z[Ensume Model Creation<br/>(Voting/Stacking)]
            Z --> AA[Final Prediction & Ranking]
            AA --> AB[Explainability Analysis<br/>(SHAP, Feature Importance)]
            AB --> AC[Visualization Generation]
            AC --> AD[Report Generation]
        '''

        # Save Mermaid diagram
        filename = "pipeline_diagram.mmd"
        filepath = os.path.join(self.assets_dir, filename)
        with open(filepath, 'w') as f:
            f.write(mermaid_code.strip())

        logger.info(f"Pipeline diagram saved to {filepath}")
        return filepath

    def generate_all_visualizations(self, feature_importance_data: Tuple[List[str], np.ndarray] = None,
                                  radar_chart_data: Tuple[Dict[str, List[float]], List[str]] = None,
                                  prediction_data: Tuple[np.ndarray, np.ndarray, List[str]] = None,
                                  waterfall_data: Tuple[List[str], np.ndarray, float, float] = None,
                                  confidence_data: Tuple[pd.DataFrame, np.ndarray, np.ndarray] = None,
                                  correlation_data: pd.DataFrame = None,
                                  model_performance_data: Dict[str, Dict[str, float]] = None) -> Dict[str, str]:
        """
        Generate all visualizations and return file paths
        """
        logger.info("Generating all visualizations...")

        file_paths = {}

        try:
            # Feature importance chart
            if feature_importance_data:
                feature_names, importances = feature_importance_data
                file_paths['feature_importance'] = self.plot_feature_importance(
                    feature_names, importances,
                    title="Feature Importance for Ballon d'Or Prediction"
                )

            # Radar chart
            if radar_chart_data:
                candidates_data, feature_names = radar_chart_data
                file_paths['radar_chart'] = self.plot_radar_chart(
                    candidates_data, feature_names,
                    title="Top 3 Ballon d'Or Candidates Comparison"
                )

            # Predicted vs actual scatter plot
            if prediction_data:
                actual_points, predicted_points, player_names = prediction_data
                file_paths['predicted_vs_actual'] = self.plot_predicted_vs_actual(
                    actual_points, predicted_points, player_names,
                    title="Actual vs Predicted Ballon d'Or Points"
                )

            # Waterfall chart
            if waterfall_data:
                feature_names, feature_values, base_value, prediction = waterfall_data
                file_paths['waterfall_chart'] = self.plot_waterfall_chart(
                    feature_names, feature_values, base_value, prediction,
                    title="Feature Contribution to Predicted Ballon d'Or Points"
                )

            # Confidence intervals
            if confidence_data:
                predictions_df, confidence_lower, confidence_upper = confidence_data
                file_paths['confidence_intervals'] = self.plot_confidence_intervals(
                    predictions_df, confidence_lower, confidence_upper,
                    title="Top 10 Ballon d'Or Predictions with 95% Confidence Intervals"
                )

            # Correlation heatmap
            if correlation_data is not None:
                file_paths['correlation_heatmap'] = self.plot_correlation_heatmap(
                    correlation_data,
                    title="Correlation Matrix of Ballon d'Or Features"
                )

            # Model comparison
            if model_performance_data:
                file_paths['model_comparison'] = self.plot_model_comparison(
                    model_performance_data,
                    title="Machine Learning Model Performance Comparison"
                )

            # Pipeline diagram
            file_paths['pipeline_diagram'] = self.create_pipeline_diagram()

            logger.info(f"Generated {len(file_paths)} visualizations")
            return file_paths

        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise


def create_sample_visualizations():
    """
    Create sample visualizations for demonstration
    """
    logger.info("Creating sample visualizations...")

    visualizer = BallonDorVisualizer()

    # Sample data for feature importance
    feature_names = [
        'goals_per_90', 'assists_per_90', 'xG_per_90', 'xA_per_90',
        'achievement_score', 'ucl_winner', 'league_title',
        'media_attention_proxy', 'social_engagement_proxy',
        'big_goals_ratio', 'clutch_factor', 'positional_multiplier'
    ]
    importances = np.array([0.15, 0.12, 0.10, 0.08, 0.18, 0.09, 0.07,
                           0.06, 0.05, 0.04, 0.03, 0.03])

    # Sample data for radar chart (top 3 candidates)
    candidates_data = {
        'Lionel Messi': [0.9, 0.8, 0.85, 0.75, 0.7, 1.0, 1.0, 0.9, 0.8, 0.85, 0.9, 1.2],
        'Erling Haaland': [0.95, 0.6, 0.9, 0.5, 0.9, 1.0, 1.0, 0.7, 0.6, 0.95, 0.8, 1.0],
        'Kylian Mbappé': [0.85, 0.85, 0.8, 0.8, 0.6, 0.0, 1.0, 0.85, 0.9, 0.7, 0.85, 1.2]
    }
    radar_features = ['Goals Rate', 'Assists Rate', 'xG Rate', 'xA Rate',
                     'Achievement Score', 'UCL Winner', 'League Title',
                     'Media Attention', 'Social Engagement', 'Big Goals',
                     'Clutch Factor', 'Positional Multiplier']

    # Sample prediction data
    np.random.seed(42)
    n_players = 20
    actual_points = np.random.uniform(10, 100, n_players)
    predicted_points = actual_points + np.random.normal(0, 10, n_players)
    predicted_points = np.maximum(predicted_points, 0)  # No negative points
    player_names = [f'Player_{i+1}' for i in range(n_players)]
    # Make first few players the top performers
    sorted_idx = np.argsort(predicted_points)[::-1]
    player_names[sorted_idx[0]] = 'Lionel Messi'
    player_names[sorted_idx[1]] = 'Erling Haaland'
    player_names[sorted_idx[2]] = 'Kylian Mbappé'

    # Sample waterfall data
    waterfall_features = ['Base Score', 'Goals_per_90', 'Assists_per_90', 'Achievement_Score',
                         'UCL_Winner', 'Media_Proxy', 'Social_Proxy']
    waterfall_values = [0, 15.2, 8.7, 12.3, 5.0, 3.2, 2.1]  # Contributions
    base_value = 25.0  # Base prediction
    final_prediction = base_value + np.sum(waterfall_values)  # 71.5

    # Sample confidence interval data
    top_10_df = pd.DataFrame({
        'player_name': ['Lionel Messi', 'Erling Haaland', 'Kylian Mbappé',
                       'Kevin De Bruyne', 'Karim Benzema', 'Mohamed Salah',
                       'Vinicius Junior', 'Jude Bellingham', 'Rodri', 'Nicolo Barella'],
        'predicted_points': [85.2, 78.5, 72.3, 65.1, 62.4, 58.9, 55.2, 52.8, 50.1, 48.7]
    })
    confidence_lower = np.array([80.1, 73.2, 67.0, 60.5, 58.0, 54.2, 51.0, 48.5, 46.0, 44.3])
    confidence_upper = np.array([90.3, 83.8, 77.6, 69.7, 66.8, 63.6, 59.4, 57.1, 54.2, 53.1])

    # Sample correlation data
    np.random.seed(123)
    corr_data = pd.DataFrame(
        np.random.randn(100, 8),
        columns=['goals_per_90', 'assists_per_90', 'xG_per_90', 'xA_per_90',
                'achievement_score', 'ucl_winner', 'media_attention', 'social_engagement']
    )
    # Add some correlations
    corr_data['xG_per_90'] = corr_data['goals_per_90'] * 0.8 + np.random.randn(100) * 0.2
    corr_data['xA_per_90'] = corr_data['assists_per_90'] * 0.7 + np.random.randn(100) * 0.3
    corr_data['achievement_score'] = (corr_data['goals_per_90'] + corr_data['assists_per_90']) * 0.5 + np.random.randn(100) * 0.5
    corr_data['ucl_winner'] = (corr_data['achievement_score'] > 0).astype(int)

    # Sample model performance data
    model_performance = {
        'RandomForest': {'train_mae': 8.2, 'train_rmse': 11.5, 'train_r2': 0.78,
                        'val_mae': 9.1, 'val_rmse': 12.3, 'val_r2': 0.72},
        'XGBoost': {'train_mae': 7.8, 'train_rmse': 10.9, 'train_r2': 0.81,
                   'val_mae': 8.5, 'val_rmse': 11.2, 'val_r2': 0.79},
        'LightGBM': {'train_mae': 7.9, 'train_rmse': 11.1, 'train_r2': 0.80,
                    'val_mae': 8.7, 'val_rmse': 11.5, 'val_r2': 0.77},
        'CatBoost': {'train_mae': 8.1, 'train_rmse': 11.3, 'train_r2': 0.79,
                    'val_mae': 8.9, 'val_rmse': 11.8, 'val_r2': 0.75},
        'Ridge': {'train_mae': 9.5, 'train_rmse': 13.2, 'train_r2': 0.68},
        'ElasticNet': {'train_mae': 9.3, 'train_rmse': 12.9, 'train_r2': 0.70}
    }

    # Generate all visualizations
    file_paths = visualizer.generate_all_visualizations(
        feature_importance_data=(feature_names, importances),
        radar_chart_data=(candidates_data, radar_features),
        prediction_data=(actual_points, predicted_points, player_names[:10]),  # Top 10 for scatter
        waterfall_data=(waterfall_features, np.array(waterfall_values), base_value, final_prediction),
        confidence_data=(top_10_df, confidence_lower, confidence_upper),
        correlation_data=corr_data,
        model_performance_data=model_performance
    )

    logger.info("Sample visualizations created:")
    for name, path in file_paths.items():
        logger.info(f"  {name}: {path}")

    return file_paths


if __name__ == "__main__":
    # Create sample visualizations
    create_sample_visualizations()
    print("Visualization module test completed!")