##--------------------------------------------------------------------##
# 1.1 Splits Analysis
##--------------------------------------------------------------------##
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from src.split_function import proportional_group_train_test_split

# Create directory for saving results
os.makedirs('split_results', exist_ok=True)

def run_split_analysis(df, group_col, target_col, n_tests=2000, test_size=0.2, save_path_full_record, save_path_unique):
    """
    Enhanced analysis that:
    1. Runs `n_tests` splits with different random states
    2. Records all group assignments
    3. Identifies unique splits with a `test_size` of 20%
    4. Saves comprehensive results
    """
    # Storage for full results
    full_results = {
        'group_assignments': pd.DataFrame(index=df[group_col].unique()),
        'split_signatures': {},
        'label_distributions': [],
        'test_sizes': []
    }

    # Track all splits and their random states
    split_records = []
    unique_splits = {}

    # Run multiple splits with different random states
    for i in tqdm(range(n_tests), desc="Running splits"):
        train_idx, test_idx = proportional_group_train_test_split(
            df, group_col, target_col,
            test_size=test_size,
            random_state=i,
            max_attempts=50,
            size_tolerance=0.05,
            divergence_threshold=0.02
        )

        # Get test groups for this split
        test_groups = set(df.loc[test_idx, group_col].unique())
        test_group_str = ','.join(sorted(test_groups))

        # Record group assignments
        full_results['group_assignments'][f'run_{i}'] = full_results['group_assignments'].index.isin(test_groups)

        # Record split signature
        if test_group_str not in unique_splits:
            unique_splits[test_group_str] = {
                'random_states': [i],
                'occurrences': 1,
                'test_groups': test_groups,
                'group_count': len(test_groups)
            }
        else:
            unique_splits[test_group_str]['random_states'].append(i)
            unique_splits[test_group_str]['occurrences'] += 1

        # Record label distribution and size
        test_dist = df.loc[test_idx, target_col].value_counts(normalize=True).to_dict()
        full_results['label_distributions'].append(test_dist)
        full_results['test_sizes'].append(len(test_idx)/len(df))

        # Store complete record
        split_records.append({
            'random_state': i,
            'test_size': len(test_idx),
            'test_ratio': len(test_idx)/len(df),
            'test_groups': test_group_str,
            'label_distribution': str(test_dist)
        })

    # Save full results
    pd.DataFrame(split_records).to_csv(save_path_full_record, index=False)

    # Save unique splits
    unique_splits_data = []
    for split_str, split_info in unique_splits.items():
        unique_splits_data.append({
            'split_id': split_str,
            'test_groups': split_str,
            'occurrences': split_info['occurrences'],
            'random_states': str(split_info['random_states']),
            'group_count': split_info['group_count']
        })

    unique_splits_df = pd.DataFrame(unique_splits_data)
    unique_splits_df.to_csv(save_path_unique, index=False)

    # Analyze results
    full_results['group_consistency'] = full_results['group_assignments'].mean(axis=1)
    full_results['label_stability'] = pd.DataFrame(full_results['label_distributions']).std()
    full_results['size_variation'] = np.std(full_results['test_sizes'])

    # Add group sizes to results
    group_sizes = df.groupby(group_col).size()
    full_results['group_sizes'] = group_sizes

    return full_results, unique_splits_df

##--------------------------------------------------------------------##
# 1.2 Evaluation & Visualization
##--------------------------------------------------------------------##

def enhanced_visualization(results, unique_splits_df, save_path_fig):
    """Expanded visualization with more insights"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # Sample 10% of results for visualization
    sample_cols = np.random.choice(
        results['group_assignments'].columns,
        size=int(len(results['group_assignments'].columns)*0.1),
        replace=False
    )

    # 1. Group assignment heatmap (sample)
    sns.heatmap(
        results['group_assignments'][sample_cols].T,
        cmap=['yellow', 'blue'],
        ax=axes[0, 0]
    )
    axes[0, 0].set_title(f'Group Assignment Sample, yellow = Train, blue = Test')

    # 2. Group consistency distribution
    sns.histplot(
        results['group_consistency'],
        bins=20,
        kde=True,
        ax=axes[0, 1]
    )
    axes[0, 1].axvline(0.5, color='red', linestyle='--')
    axes[0, 1].set_title('Group Assignment Consistency Distribution')

    # 3. Label distribution stability
    results['label_stability'].plot(
        kind='bar',
        ax=axes[1, 0],
        color='orange'
    )
    axes[1, 0].set_title('Label Distribution Stability')

    # 4. Test size distribution
    axes[1, 1].hist(results['test_sizes'], bins=20)
    axes[1, 1].axvline(np.mean(results['test_sizes']), color='red')
    axes[1, 1].set_title(f'Test Size Variation (σ={results["size_variation"]:.4f})')

    # 5. Group size vs test frequency
    sns.regplot(
        x=results['group_sizes'],
        y=results['group_consistency'],
        ax=axes[2, 0],
        scatter_kws={'alpha': 0.6}
    )
    axes[2, 0].set_title('Group Size vs. Test Frequency')

    # 6. Unique splits summary
    axes[2, 1].bar(
        ['Unique Splits', 'Total Runs'],
        [len(unique_splits_df), len(results['group_assignments'].columns)]
    )
    axes[2, 1].set_title('Split Variability')

    plt.tight_layout()
    plt.savefig(save_path_fig, dpi=300)
    plt.close()
