##--------------------------------------------------------------------##
# 2.1 Feature Distribution Analysis
##--------------------------------------------------------------------##
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from src.split_function import proportional_group_train_test_split

def calculate_distribution_differences(
    df,
    seeds,
    group_col='Source', # column name for grouping
    target_col='y_var', # y variable name
    test_size=0.2, # testing set ratio
    **split_kwargs
):
    """
    Calculate distribution differences for features across multiple data splits.

    For each seed:
    1. Split data using proportional_group_train_test_split
    2. Compute distribution differences for each feature between train/test sets
    3. Return per-seed results for detailed analysis

    Parameters:
        df (pd.DataFrame): Input dataframe
        seeds (list): List of random seeds to generate splits
        group_col (str): Column name for grouping
        target_col (str): Target variable column
        test_size (float): Size of test split
        **split_kwargs: Additional arguments for proportional_group_train_test_split

    Returns:
        pd.DataFrame: Per-seed distribution difference metrics
    """
    # Identify features to analyze
    features = [col for col in df.columns if col not in [group_col, target_col]]
    if not features:
        raise ValueError("No features found for analysis")

    # Precompute feature types accounting for one-hot encoding
    feature_types = {}
    for feature in features:
        # Skip constant features
        if df[feature].nunique() <= 1:
            feature_types[feature] = 'constant'
            continue

        if (pd.api.types.is_numeric_dtype(df[feature]) and
            df[feature].dropna().isin([0, 1]).all() and
            df[feature].nunique() == 2):
            feature_types[feature] = 'categorical'
        elif pd.api.types.is_numeric_dtype(df[feature]):
            feature_types[feature] = 'numeric'
        else:
            feature_types[feature] = 'categorical'

    # Initialize results storage
    results = []

    for seed in seeds:
        # Create data split
        train_idx, test_idx = proportional_group_train_test_split(
            df,
            group_col=group_col,
            target_col=target_col,
            test_size=test_size,
            random_state=seed,
            **split_kwargs
        )

        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        # Calculate feature distribution differences
        for feature in features:
            # Skip constant features
            if feature_types[feature] == 'constant':
                continue

            if feature_types[feature] == 'numeric':
                # Handle missing values
                train_vals = train_df[feature].dropna()
                test_vals = test_df[feature].dropna()

                # Skip if insufficient data
                if len(train_vals) < 2 or len(test_vals) < 2:
                    continue

                # Kolmogorov-Smirnov distance
                ks_stat, _ = ks_2samp(train_vals, test_vals)

                # Wasserstein distance
                wasser_dist = wasserstein_distance(train_vals, test_vals)

                # Normalized Wasserstein (scaled by IQR)
                iqr = np.subtract(*np.percentile(train_vals, [75, 25]))
                norm_wasser = wasser_dist / iqr if iqr > 0 else np.nan

                results.append({
                    'seed': seed,
                    'feature': feature,
                    'type': feature_types[feature],
                    'KS': ks_stat,
                    'Wasserstein': wasser_dist,
                    'Norm_Wasserstein': norm_wasser
                })
            else:  # Categorical features (including one-hot encoded)
                # Treat missing values as separate category
                train_vals = train_df[feature].fillna('MISSING').astype(str)
                test_vals = test_df[feature].fillna('MISSING').astype(str)

                # Get all categories
                all_cats = sorted(set(train_vals) | set(test_vals))

                # Create probability distributions
                train_counts = train_vals.value_counts().reindex(all_cats, fill_value=0)
                test_counts = test_vals.value_counts().reindex(all_cats, fill_value=0)

                p_train = train_counts / len(train_vals)
                p_test = test_counts / len(test_vals)

                # Total Variation Distance
                tvd = 0.5 * np.sum(np.abs(p_train - p_test))

                # Jensen-Shannon Divergence
                js_div = jensenshannon(p_train, p_test, base=2)  # Returns sqrt(JS_div)
                js_div = js_div ** 2  # Convert to true JS divergence [0, 1]

                results.append({
                    'seed': seed,
                    'feature': feature,
                    'type': feature_types[feature],
                    'TVD': tvd,
                    'JS_divergence': js_div
                })

    # Return per-seed results without aggregation
    return pd.DataFrame(results)

##--------------------------------------------------------------------##
# 2.2 Results Summary
##--------------------------------------------------------------------##

# Create summary across seeds
def summarize_by_seed(per_seed_results):
    """
    Aggregate distribution differences by seed across all features.

    Parameters:
        per_seed_results (pd.DataFrame): Output from calculate_distribution_differences

    Returns:
        pd.DataFrame: Per-seed summary of distribution differences
    """
    # Create a copy to avoid modifying original data
    df = per_seed_results.copy()

    # Create a unified divergence metric:
    # For numeric: Normalized Wasserstein (scaled by IQR) or KS if normalized is missing
    # For categorical: JS Divergence
    df['unified_divergence'] = np.where(
        df['type'] == 'numeric',
        np.where(pd.notna(df['Norm_Wasserstein']),
                 df['Norm_Wasserstein'],
                 df['KS']),
        df['JS_divergence']
    )

    # Aggregate metrics by seed
    seed_summary = df.groupby('seed').agg(
        num_features=('feature', 'count'),
        num_numeric=('type', lambda x: (x == 'numeric').sum()),
        num_categorical=('type', lambda x: (x == 'categorical').sum()),
        avg_KS=('KS', 'mean'),
        avg_Wasserstein=('Wasserstein', 'mean'),
        avg_Norm_Wasserstein=('Norm_Wasserstein', 'mean'),
        avg_TVD=('TVD', 'mean'),
        avg_JS_divergence=('JS_divergence', 'mean'),
        min_divergence=('unified_divergence', 'min'),
        mean_divergence=('unified_divergence', 'mean'),
        median_divergence=('unified_divergence', 'median'),
        max_divergence=('unified_divergence', 'max'),
        std_divergence=('unified_divergence', 'std')
    ).reset_index()

    # Calculate divergence score (lower is better)
    seed_summary['divergence_score'] = seed_summary['mean_divergence'] / seed_summary['std_divergence'].clip(lower=0.1)

    return seed_summary.sort_values(by='mean_divergence')

##--------------------------------------------------------------------##
# 2.3 Good Seed Identification
##--------------------------------------------------------------------##
import pandas as pd
import numpy as np

def select_good_seeds(per_seed_results,
                      numeric_threshold=0.05,
                      categorical_threshold=0.08,
                      min_pass_rate=0.9):
    """
    Select seeds where ≥90% of features meet distribution preservation thresholds

    Parameters:
        per_seed_results (pd.DataFrame): Output from calculate_distribution_differences
        numeric_threshold (float): Max acceptable normalized Wasserstein distance
        categorical_threshold (float): Max acceptable JS divergence
        min_pass_rate (float): Minimum % of features passing thresholds (0.9 = 90%)

    Returns:
        pd.DataFrame: Quality report for all seeds
        list: Seeds meeting quality criteria
    """
    # Create a copy to avoid modifying original data
    df = per_seed_results.copy()

    # Create pass/fail indicators
    df['pass_numeric'] = (
        (df['type'] == 'numeric') &
        (df['Wasserstein'] <= numeric_threshold)
    )

    df['pass_categorical'] = (
        (df['type'] == 'categorical') &
        (df['JS_divergence'] <= categorical_threshold)
    )

    # Combine into overall pass indicator
    df['pass_feature'] = np.where(
        df['type'] == 'numeric',
        df['pass_numeric'],
        df['pass_categorical']
    )

    # Calculate per-seed metrics
    seed_quality = df.groupby('seed').agg(
        num_features=('feature', 'count'),
        num_numeric=('type', lambda x: (x == 'numeric').sum()),
        num_categorical=('type', lambda x: (x == 'categorical').sum()),
        num_pass=('pass_feature', 'sum'),
        pass_rate=('pass_feature', 'mean'),

        # Max divergence metrics
        max_numeric_divergence=('Wasserstein', 'max'),
        max_categorical_divergence=('JS_divergence', 'max'),

        # Mean divergence metrics
        mean_numeric_divergence=('Wasserstein', 'mean'),
        mean_categorical_divergence=('JS_divergence', 'mean')
    ).reset_index()

    # Add quality flag
    seed_quality['good_seed'] = seed_quality['pass_rate'] >= min_pass_rate

    # Sort by quality
    seed_quality = seed_quality.sort_values(
        by=['pass_rate', 'max_numeric_divergence', 'max_categorical_divergence'],
        ascending=[False, True, True]
    )

    # Extract good seeds
    good_seeds = seed_quality[seed_quality['good_seed']]['seed'].tolist()

    return seed_quality, good_seeds
