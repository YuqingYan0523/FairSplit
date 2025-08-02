##--------------------------------------------------------------------##
# 3.1 Example Data Loading
##--------------------------------------------------------------------##
import os
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, 'example_dataset.csv')
df = pd.read_csv(DATA_PATH)

##--------------------------------------------------------------------##
# 3.2 Evaluation of Seeds Ensuring y Variable Distribution
##--------------------------------------------------------------------##
from src.initial_screening import run_split_analysis, enhanced_visualization

if __name__ == "__main__":
    FR_CSV = os.path.join(CURRENT_DIR, 'full_split_records.csv') #this csv spreadsheet...
    UQ_CSV = os.path.join(CURRENT_DIR, 'unique_splits.csv') #this csv spreadsheet...
    # Run analysis
    full_results, unique_splits_df = run_split_analysis(
        df,
        group_col='Source', #your literature source
        target_col='y_var', #year of the study, a common important indictor; change it to any other indictor if needs
        n_tests=2000, #running a test of 2000 different seeds; change the number as needed
        test_size=0.2, #a common ratio 8:2 is used
        save_path_full_record = FR_CSV,
        save_path_unique = UQ_CSV
    )

    # Generate enhanced visualization
    FIG_PATH = os.path.join(CURRENT_DIR, 'split_analysis.png')
    enhanced_visualization(full_results, unique_splits_df, FIG_PATH)

    print(f"Total unique splits found: {len(unique_splits_df)}")
    print(f"Most common split occurred {unique_splits_df['occurrences'].max()} times")

##--------------------------------------------------------------------##
# 3.3 Extract and Analyze Good Seeds from Last Step
##--------------------------------------------------------------------##
from src.refined_screening import calculate_distribution_differences, summarize_by_seed, select_good_seeds

# Random states from seed controlling
US_PATH = os.path.join(CURRENT_DIR, 'unique_splits.csv')
unique_splits_df = pd.read_csv(US_PATH)

def extract_max_state(random_states_str):
    # Remove brackets and split by commas
    try:
        numbers = [int(x.strip()) for x in random_states_str.strip('[]').split(',')]
        return max(numbers) if numbers else None
    except:
        return None

# Apply the function and find overall maximum
unique_splits_df['max_random_state'] = unique_splits_df['random_states'].apply(extract_max_state)
random_states = unique_splits_df['max_random_state']

diff_results = calculate_distribution_differences(
    df=df,
    seeds=random_states,
    group_col='Source',
    target_col='y_var',
    test_size=0.2,
    max_attempts=50, #the number of attempts that is used to search a valid result
    size_tolerance=0.05, # 
    divergence_threshold=0.02 #
)

DR_CSV = os.path.join(CURRENT_DIR, 'distribution_differences.csv')
diff_results.to_csv(DR_CSV, index=False)

# Create summary statistics across features
summary_stats = diff_results.groupby(['feature', 'type']).agg({
    'KS': ['mean', 'std'],
    'Wasserstein': ['mean', 'std'],
    'Norm_Wasserstein': ['mean', 'std'],
    'TVD': ['mean', 'std'],
    'JS_divergence': ['mean', 'std']
}).reset_index()

# Flatten multi-level columns
summary_stats.columns = [
    f'{col[0]}_{col[1]}' if col[1] != '' else col[0]
    for col in summary_stats.columns
]

# Rename columns for clarity
summary_stats = summary_stats.rename(columns={
    'feature_': 'feature',
    'type_': 'type'
})

# Show features with highest average JS divergence
summary_stats.to_csv('distribution_summary.csv', index=False)

seed_summary = summarize_by_seed(diff_results)
SS_CSV = os.path.join(CURRENT_DIR, 'seed_summary.csv')
seed_summary.to_csv(SS_CSV, index=False)

##--------------------------------------------------------------------##
# 3.4 Identifying Good Seeds Ensuring X Distribution
##--------------------------------------------------------------------##
seed_quality, good_seeds = select_good_seeds(diff_results)
SQ_CSV = os.path.join(CURRENT_DIR, 'seed_quality.csv')
seed_quality.to_csv(SQ_CSV, index=False)
