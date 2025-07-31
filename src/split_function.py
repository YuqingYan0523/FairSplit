##--------------------------------------------------------------------##
# 0. A grouped data splitting function
##--------------------------------------------------------------------##
from collections import defaultdict
import pandas as pd

def proportional_group_train_test_split(
    df,
    group_col='Source',
    target_col='y_var',
    test_size=0.2,
    random_state=42,
    max_attempts=50,
    size_tolerance=0.05,
    divergence_threshold=0.02
):
    """
    Final corrected version with proper random state handling
    """
    # Calculate targets
    total_rows = len(df)
    target_test_rows = int(total_rows * test_size)
    min_test_rows = int(target_test_rows * (1 - size_tolerance))
    max_test_rows = int(target_test_rows * (1 + size_tolerance))

    # Get overall label distribution
    if pd.api.types.is_numeric_dtype(df[target_col]):
        overall_label_dist = df[target_col].mean()  
        is_numeric_target = True
    else:
        overall_label_dist = df[target_col].value_counts(normalize=True).to_dict()
        labels = sorted(overall_label_dist.keys())
        is_numeric_target = False

    # Precompute group statistics
    group_data = []
    for name, group in df.groupby(group_col):
        label_counts = group[target_col].value_counts().to_dict()
        group_data.append({
            'name': name,
            'size': len(group),
            'label_counts': label_counts,
            'label_dist': {k: v/len(group) for k,v in label_counts.items()}
        })
    groups_df = pd.DataFrame(group_data)

    # Scoring function
    def evaluate_split(selected_groups):
        test_mask = df[group_col].isin(selected_groups)
        test_size = test_mask.sum()

        if test_size == 0:
            return float('inf'), float('inf')

        if is_numeric_target:
            test_mean = df.loc[test_mask, target_col].mean()
            divergence = abs(test_mean - overall_label_dist) / (abs(overall_label_dist) + 1e-8)  # Relative error as the divergence for regression model
        else:
            test_dist = df.loc[test_mask, target_col].value_counts(normalize=True).to_dict()
            divergence = sum(abs(test_dist.get(l, 0) - overall_label_dist[l]) for l in labels)  # Relative ratios of each label as the divergence metric for classification model
        
        size_penalty = ((test_size - target_test_rows)/target_test_rows)**2
        return divergence, size_penalty

    # Optimization with proper random state handling
    best_split = None
    best_score = float('inf')

    for attempt in range(max_attempts):
        # Create new random state for each attempt
        attempt_rng = np.random.RandomState(random_state + attempt)
        shuffled = groups_df.sample(frac=1, random_state=attempt_rng)

        # Greedy accumulation
        test_groups = []
        current_size = 0
        for _, group in shuffled.iterrows():
            if current_size + group['size'] > max_test_rows:
                continue

            test_groups.append(group['name'])
            current_size += group['size']

            if current_size >= min_test_rows:
                break

        # Evaluate
        divergence, size_penalty = evaluate_split(test_groups)
        score = divergence + 10*size_penalty

        if score < best_score:
            best_score = score
            best_split = {
                'test_groups': test_groups,
                'test_size': current_size,
                'divergence': divergence,
                'attempt': attempt + 1
            }

            # Early exit for perfect match
            if (divergence <= divergence_threshold and
                abs(current_size - target_test_rows)/target_test_rows <= size_tolerance):
                break

    # Fallback handling
    if best_split is None:
        raise ValueError(f"No valid split found in {max_attempts} attempts. Try increasing size_tolerance or max_attempts.")

    # Create splits
    test_idx = df[df[group_col].isin(best_split['test_groups'])].index
    train_idx = df.index.difference(test_idx)

    # Diagnostic output
    print(f"Best split found after {best_split['attempt']} attempts:")
    print(f"- Test size: {len(test_idx)} ({len(test_idx)/total_rows:.1%} of total)")
    print(f"- Label divergence: {best_split['divergence']:.4f} (target ≤ {divergence_threshold})")
    print(f"- Size deviation: {(len(test_idx) - target_test_rows)/target_test_rows:.1%} (allowed ±{size_tolerance:.0%})")

    return train_idx, test_idx