"""
Feature engineering for fraud detection.
Creates temporal, behavioral, and contextual features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder


class FeatureEngineerer:
    """
    Engineer features for fraud detection model.

    Creates:
    - Temporal features (hour, day, weekend)
    - Behavioral features (amount deviations, velocity)
    - Encoded categorical features
    - Scaled numerical features
    """

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Apply feature engineering to dataset.

        Args:
            df: Raw transaction data
            fit: Whether to fit scalers/encoders (True for train, False for test)

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Temporal features (already in data, just ensure they exist)
        df = self._ensure_temporal_features(df)

        # Behavioral features
        df = self._create_behavioral_features(df)

        # Encode categorical features
        df = self._encode_categorical_features(df, fit=fit)

        # Scale numerical features
        df = self._scale_numerical_features(df, fit=fit)

        if fit:
            self.fitted = True

        return df

    def _ensure_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure temporal features exist."""
        if 'hour' not in df.columns and 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour

        if 'day_of_week' not in df.columns and 'timestamp' in df.columns:
            df['day_of_week'] = df['timestamp'].dt.dayofweek

        if 'is_weekend' not in df.columns:
            if 'day_of_week' in df.columns:
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            else:
                df['is_weekend'] = 0

        return df

    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral and risk-indicating features."""
        # Amount-based features
        if 'amount' in df.columns:
            # Log amount (handles skewness)
            df['log_amount'] = np.log1p(df['amount'])

            # Amount deviation from mean (per merchant category)
            if 'merchant_category' in df.columns:
                merchant_means = df.groupby('merchant_category')['amount'].transform('mean')
                df['amount_vs_merchant_avg'] = df['amount'] / (merchant_means + 1)

            # High amount indicator
            df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.90)).astype(int)

            # Very high amount (top 1%)
            df['is_very_high_amount'] = (df['amount'] > df['amount'].quantile(0.99)).astype(int)

        # Risk indicators based on entry mode
        if 'entry_mode' in df.columns:
            df['is_risky_entry_mode'] = df['entry_mode'].isin(
                ['magnetic_stripe', 'online']
            ).astype(int)

        # International transaction risk
        if 'is_international' in df.columns:
            pass  # Already binary

        # Card type risk
        if 'card_type' in df.columns:
            df['is_credit_card'] = (df['card_type'] == 'credit').astype(int)

        # Unusual time (late night/early morning)
        if 'hour' in df.columns:
            df['is_unusual_time'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)

        # Customer age-based risk
        if 'customer_age' in df.columns:
            df['age_group'] = pd.cut(
                df['customer_age'],
                bins=[0, 25, 35, 50, 65, 100],
                labels=['18-25', '26-35', '36-50', '51-65', '65+']
            ).astype(str)

            # Young adults have higher fraud risk
            df['is_young_adult'] = (df['customer_age'] < 30).astype(int)

        return df

    def _encode_categorical_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """Encode categorical features."""
        categorical_cols = [
            'merchant_category',
            'card_type',
            'entry_mode',
            'region',
            'time_of_day',
            'age_group'
        ]

        # Only encode columns that exist
        categorical_cols = [col for col in categorical_cols if col in df.columns]

        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()

            if fit:
                # Handle unseen categories by fitting with all possible values
                df[col] = df[col].astype(str)
                df[col + '_encoded'] = self.encoders[col].fit_transform(df[col])
            else:
                # For test data, handle unseen categories
                df[col] = df[col].astype(str)
                # Map unseen categories to -1, then add 1 to shift all values
                encoded = df[col].map(
                    lambda x: self.encoders[col].classes_.tolist().index(x)
                    if x in self.encoders[col].classes_ else -1
                )
                df[col + '_encoded'] = encoded + 1  # Shift so -1 becomes 0

        return df

    def _scale_numerical_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """Scale numerical features."""
        numerical_cols = [
            'amount',
            'log_amount',
            'amount_vs_merchant_avg',
            'customer_age',
            'customer_income',
            'hour',
            'day_of_week'
        ]

        # Only scale columns that exist
        numerical_cols = [col for col in numerical_cols if col in df.columns]

        for col in numerical_cols:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()

            if fit:
                df[col + '_scaled'] = self.scalers[col].fit_transform(
                    df[[col]].values.reshape(-1, 1)
                ).flatten()
            else:
                df[col + '_scaled'] = self.scalers[col].transform(
                    df[[col]].values.reshape(-1, 1)
                ).flatten()

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns for model input.

        Args:
            df: DataFrame after feature engineering

        Returns:
            List of feature column names
        """
        # Get all encoded and scaled columns
        feature_cols = []

        # Encoded categorical features
        for col in df.columns:
            if col.endswith('_encoded'):
                feature_cols.append(col)

        # Scaled numerical features
        for col in df.columns:
            if col.endswith('_scaled'):
                feature_cols.append(col)

        # Binary indicators
        binary_cols = [
            'is_weekend',
            'is_high_amount',
            'is_very_high_amount',
            'is_risky_entry_mode',
            'is_international',
            'is_credit_card',
            'is_unusual_time',
            'is_young_adult'
        ]
        for col in binary_cols:
            if col in df.columns:
                feature_cols.append(col)

        return sorted(feature_cols)

    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted scalers/encoders.

        Args:
            df: New transaction data

        Returns:
            DataFrame with engineered features
        """
        if not self.fitted:
            raise ValueError("FeatureEngineerer must be fitted before transform_new_data")

        return self.fit_transform(df, fit=False)
