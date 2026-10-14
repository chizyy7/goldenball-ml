"""
Feature Engineering Module for Ballon d'Or Prediction
Creates features across three weighted categories:
A. Individual Performance
B. Collective Achievements
C. Contextual / Narrative
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BallonDorFeatureEngineer:
    """Engineers features for Ballon d'Or prediction model"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.positional_multipliers = {
            'Forward': 1.2,
            'Midfielder': 1.0,
            'Defender': 0.9,
            'Goalkeeper': 0.7
        }
        self.league_strength = {
            'Premier League': 1.15,
            'La Liga': 1.1,
            'Bundesliga': 1.05,
            'Serie A': 1.05,
            'Ligue 1': 1.0,
            'MLS': 0.8,
            'Other': 0.9
        }

    def create_individual_performance_features(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create individual performance features (Category A)
        Features: goals, assists, xG, xA, per-90 rates, big game performance, etc.
        """
        logger.info("Creating individual performance features...")

        df = stats_df.copy()

        # Ensure required columns exist with defaults if missing
        if 'position' not in df.columns:
            df['position'] = 'Forward'  # Default position
        if 'league' not in df.columns:
            df['league'] = 'Other'  # Default league

        # Basic per-90 features
        df['goals_per_90'] = df['goals'] / (df['minutes_played'] / 90)
        df['assists_per_90'] = df['assists'] / (df['minutes_played'] / 90)
        df['xG_per_90'] = df['xG'] / (df['minutes_played'] / 90)
        df['xA_per_90'] = df['xA'] / (df['minutes_played'] / 90)
        df['goal_plus_assist_per_90'] = df['goals_per_90'] + df['assists_per_90']
        df['xG_plus_xA_per_90'] = df['xG_per_90'] + df['xA_per_90']

        # Shot creation and advanced metrics (estimated from available data)
        df['shot_creating_actions'] = df['assists'] * 2  # Placeholder (using assists as proxy)
        df['progressive_carries_estimated'] = df['minutes_played'] * 0.1  # Placeholder

        # Position-specific adjustments
        df['positional_multiplier'] = df['position'].map(self.positional_multipliers).fillna(1.0)
        df['adjusted_goals_per_90'] = df['goals_per_90'] * df['positional_multiplier']
        df['adjusted_xG_per_90'] = df['xG_per_90'] * df['positional_multiplier']

        # League strength adjustment
        df['league_multiplier'] = df['league'].map(self.league_strength).fillna(1.0)
        df['league_adjusted_goals_per_90'] = df['goals_per_90'] * df['league_multiplier']
        df['league_adjusted_xG_per_90'] = df['xG_per_90'] * df['league_multiplier']

        # Big game performance indicators (estimated)
        df['big_goals_ratio'] = np.minimum(df['goals'] * 0.3 / df['matches_played'], 1.0)  # Estimated
        df['clutch_factor'] = np.minimum((df['goals'] + df['assists']) * 0.2 / df['matches_played'], 1.0)

        # Consistency and form
        df['goals_per_match'] = df['goals'] / df['matches_played']
        df['assists_per_match'] = df['assists'] / df['matches_played']
        df['goal_involvement_per_match'] = df['goals_per_match'] + df['assists_per_match']

        # Efficiency metrics
        df['goals_per_xG'] = np.where(df['xG'] > 0, df['goals'] / df['xG'], 0)
        df['assists_per_xA'] = np.where(df['xA'] > 0, df['assists'] / df['xA'], 0)

        # Select and return key individual performance features
        individual_features = [
            'goals_per_90', 'assists_per_90', 'xG_per_90', 'xA_per_90',
            'goal_plus_assist_per_90', 'xG_plus_xA_per_90',
            'adjusted_goals_per_90', 'adjusted_xG_per_90',
            'league_adjusted_goals_per_90', 'league_adjusted_xG_per_90',
            'shot_creating_actions', 'progressive_carries_estimated',
            'big_goals_ratio', 'clutch_factor',
            'goals_per_match', 'assists_per_match', 'goal_involvement_per_match',
            'goals_per_xG', 'assists_per_xA'
        ]

        # Filter to only existing columns
        available_features = [f for f in individual_features if f in df.columns]
        logger.info(f"Created {len(available_features)} individual performance features")

        return df[['year', 'player_name', 'team'] + available_features]

    def create_collective_achievement_features(self, achievements_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create collective achievement features (Category B)
        Features: UCL winner, league titles, domestic cups, international trophies, etc.
        """
        logger.info("Creating collective achievement features...")

        df = achievements_df.copy()

        # Binary achievement features
        df['ucl_winner'] = df['ucl_winner'].astype(int)
        df['league_title'] = df['league_title'].astype(int)
        df['domestic_cup_winner'] = df['domestic_cup'].astype(int)
        df['international_trophy_winner'] = df['international_trophy'].astype(int)
        df['treble_winner'] = df['treble_double'].apply(lambda x: 1 if x == True else 0)  # Simplified
        df['double_winner'] = df['treble_double'].apply(lambda x: 1 if x == True else 0)  # Simplified

        # Achievement scoring system
        df['achievement_score'] = (
            df['ucl_winner'] * 3 +           # Champions League = 3 points
            df['league_title'] * 2 +         # League title = 2 points
            df['domestic_cup_winner'] * 1 +  # Domestic cup = 1 point
            df['international_trophy_winner'] * 2 +  # International trophy = 2 points
            df['treble_winner'] * 4 +        # Treble = 4 points
            df['double_winner'] * 2          # Double = 2 points
        )

        # Team dominance weighting (based on achievement score)
        df['team_dominance_weight'] = np.minimum(df['achievement_score'] / 10, 1.0)

        # Player contribution to team success (estimated)
        df['contribution_to_success'] = df['achievement_score'] * 0.1  # Placeholder

        # Select and return key collective achievement features
        collective_features = [
            'ucl_winner', 'league_title', 'domestic_cup_winner',
            'international_trophy_winner', 'treble_winner', 'double_winner',
            'achievement_score', 'team_dominance_weight', 'contribution_to_success'
        ]

        logger.info(f"Created {len(collective_features)} collective achievement features")
        return df[['year', 'player_name', 'team'] + collective_features]

    def create_contextual_features(self, base_df: pd.DataFrame, contextual_data: Dict = None) -> pd.DataFrame:
        """
        Create contextual/narrative features (Category C)
        Features: media mentions, social engagement, disciplinary record, etc.
        """
        logger.info("Creating contextual/narrative features...")

        df = base_df.copy()

        # Since we don't have real contextual data sources, we'll create proxies
        # based on available data and reasonable assumptions

        # Media attention proxy (based on performance)
        df['media_attention_proxy'] = (
            df['goals_per_90'] * 0.3 +
            df['assists_per_90'] * 0.2 +
            df['big_goals_ratio'] * 0.2 +
            df['clutch_factor'] * 0.2 +
            df['shot_creating_actions'] / df['shot_creating_actions'].max() * 0.1 if df['shot_creating_actions'].max() > 0 else 0
        )

        # Social media engagement proxy (based on fame and performance)
        df['social_engagement_proxy'] = np.minimum(
            (df['goals_per_90'] + df['assists_per_90']) * 2 +
            df['shot_creating_actions'] / df['shot_creating_actions'].max() * 0.5 if df['shot_creating_actions'].max() > 0 else 0, 10
        )

        # Disciplinary record / fair play proxy (inverse - fewer cards = better score)
        # Since we don't have card data, we'll assume top players have good discipline
        # Use goals_per_90 as a proxy for prominence (more prominent players might get more cards, but we assume they are disciplined)
        df['fair_play_score'] = np.maximum(5 - (df['goals_per_90'] * 0.1), 1)
        df['disciplinary_penalty'] = np.maximum(0, (df['goals_per_90'] - 5) * 0.1)  # Penalty for high scoring (as proxy for prominence)

        # Expert commentary proxy (based on performance consistency)
        df['expert_commentary_proxy'] = (
            df['goals_per_xG'] * 0.3 +
            df['assists_per_xA'] * 0.2 +
            df['goal_involvement_per_match'] * 0.3 +
            (2 - abs(df['goals_per_xG'] - 1)) * 0.2  # Closer to 1.0 is better
        )

        # Narrative momentum (recent form indicator)
        df['narrative_momentum'] = np.minimum(
            df['goals_per_match'] * 2 +
            df['assists_per_match'] * 1.5 +
            df['shot_creating_actions'] / df['shot_creating_actions'].max() * 0.5 if df['shot_creating_actions'].max() > 0 else 0, 5
        )

        # Big stage performance (clutch games)
        df['big_stage_performance'] = (
            df['big_goals_ratio'] * 0.4 +
            df['clutch_factor'] * 0.3 +
            df['shot_creating_actions'] / df['shot_creating_actions'].max() * 0.2 if df['shot_creating_actions'].max() > 0 else 0
        )

        # Select and return key contextual features
        contextual_features = [
            'media_attention_proxy', 'social_engagement_proxy',
            'fair_play_score', 'disciplinary_penalty',
            'expert_commentary_proxy', 'narrative_momentum',
            'big_stage_performance'
        ]

        logger.info(f"Created {len(contextual_features)} contextual/narrative features")
        return df[['year', 'player_name', 'team'] + contextual_features]

    def combine_all_features(self, individual_df: pd.DataFrame,
                           collective_df: pd.DataFrame,
                           contextual_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine all feature sets into a single feature matrix
        """
        logger.info("Combining all feature sets...")

        # Merge dataframes on year, player_name, team
        combined = individual_df.merge(collective_df, on=['year', 'player_name', 'team'], how='inner')
        combined = combined.merge(contextual_df, on=['year', 'player_name', 'team'], how='inner')

        logger.info(f"Combined features shape: {combined.shape}")
        return combined

    def prepare_model_features(self, feature_df: pd.DataFrame,
                             target_col: str = 'points',
                             voting_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare final feature matrix and target vector for modeling
        """
        logger.info("Preparing final features for modeling...")

        # If we have voting data, merge to get target variable
        if voting_df is not None and not voting_df.empty:
            # Aggregate voting data to get points per player per year
            voting_agg = voting_df.groupby(['year', 'player_name'])['points'].sum().reset_index()
            voting_agg = voting_agg.rename(columns={'points': 'target_points'})

            feature_df = feature_df.merge(voting_agg, on=['year', 'player_name'], how='left')
            target_series = feature_df['target_points']

            # Drop non-feature columns
            feature_cols = [col for col in feature_df.columns
                          if col not in ['year', 'player_name', 'team', 'target_points']]
            feature_matrix = feature_df[feature_cols].fillna(0)
        else:
            # If no voting data, we'll use points from feature_df if available
            if 'points' in feature_df.columns:
                target_series = feature_df['points']
                feature_cols = [col for col in feature_df.columns
                              if col not in ['year', 'player_name', 'team', 'points']]
                feature_matrix = feature_df[feature_cols].fillna(0)
            else:
                # Create a placeholder target for development
                logger.warning("No target variable found, creating placeholder")
                target_series = pd.Series(np.random.rand(len(feature_df)) * 100, index=feature_df.index)
                feature_cols = [col for col in feature_df.columns
                              if col not in ['year', 'player_name', 'team']]
                feature_matrix = feature_df[feature_cols].fillna(0)

        # Drop rows where target is NaN (if any)
        if target_series.isna().any():
            logger.warning(f"Dropping {target_series.isna().sum()} rows with NaN target")
            valid_idx = ~target_series.isna()
            feature_matrix = feature_matrix[valid_idx]
            target_series = target_series[valid_idx]

        logger.info(f"Feature matrix shape: {feature_matrix.shape}")
        logger.info(f"Target vector shape: {target_series.shape}")

        return feature_matrix, target_series

    def get_feature_importance_names(self) -> List[str]:
        """
        Return list of feature names for interpretability
        """
        return [
            # Individual Performance
            'goals_per_90', 'assists_per_90', 'xG_per_90', 'xA_per_90',
            'goal_plus_assist_per_90', 'xG_plus_xA_per_90',
            'adjusted_goals_per_90', 'adjusted_xG_per_90',
            'league_adjusted_goals_per_90', 'league_adjusted_xG_per_90',
            'shot_creating_actions', 'progressive_carries_estimated',
            'big_goals_ratio', 'clutch_factor',
            'goals_per_match', 'assists_per_match', 'goal_involvement_per_match',
            'goals_per_xG', 'assists_per_xA',

            # Collective Achievements
            'ucl_winner', 'league_title', 'domestic_cup_winner',
            'international_trophy_winner', 'treble_winner', 'double_winner',
            'achievement_score', 'team_dominance_weight', 'contribution_to_success',

            # Contextual/Narrative
            'media_attention_proxy', 'social_engagement_proxy',
            'fair_play_score', 'disciplinary_penalty',
            'expert_commentary_proxy', 'narrative_momentum',
            'big_stage_performance'
        ]


if __name__ == "__main__":
    # Example usage
    engineer = BallonDorFeatureEngineer()

    # Create sample data for testing
    sample_stats = pd.DataFrame({
        'year': [2023, 2023, 2023],
        'player_name': ['Lionel Messi', 'Erling Haaland', 'Kylian Mbappé'],
        'team': ['PSG', 'Manchester City', 'PSG'],
        'league': ['Ligue 1', 'Premier League', 'Ligue 1'],
        'position': ['Forward', 'Forward', 'Forward'],
        'matches_played': [35, 35, 42],
        'goals': [32, 36, 40],
        'assists': [15, 8, 15],
        'xG': [28.5, 34.2, 38.7],
        'xA': [18.2, 7.1, 14.8],
        'minutes_played': [2980, 2950, 3420]
    })

    sample_achievements = pd.DataFrame({
        'year': [2023, 2023, 2023],
        'player_name': ['Lionel Messi', 'Erling Haaland', 'Kylian Mbappé'],
        'team': ['PSG', 'Manchester City', 'PSG'],
        'ucl_winner': [False, True, False],
        'league_title': [True, True, True],
        'domestic_cup': [False, True, False],
        'international_trophy': [False, False, False],
        'treble_double': [False, True, False]
    })

    # Create features
    individual_features = engineer.create_individual_performance_features(sample_stats)
    collective_features = engineer.create_collective_achievement_features(sample_achievements)
    contextual_features = engineer.create_contextual_features(individual_features)

    # Combine features
    combined_features = engineer.combine_all_features(
        individual_features, collective_features, contextual_features
    )

    print("Sample feature engineering completed:")
    print(f"Individual features shape: {individual_features.shape}")
    print(f"Collective features shape: {collective_features.shape}")
    print(f"Contextual features shape: {contextual_features.shape}")
    print(f"Combined features shape: {combined_features.shape}")
    print(f"Feature columns: {list(combined_features.columns)}")