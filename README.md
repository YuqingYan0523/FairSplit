# FairSplit: A Data Splitting & Randomness Management Tool for Small, Heterogeneous and Imbalanced Datasets' Utilization in Machine Learing 
Authors: [Yuqing Yan](https://github.com/YuqingYan0523) and [Junjie Zhu](https://github.com/starfriend10) @Princeton University WET Lab led by Z. Jason Ren

## Work Origin and Citation
You can find and cite the original study here:

*<ins>Publication DOI reserved</ins>*

## Pipeline Overview
1. Motivation & Backgrounds
This repository was developed as part of a research project predicting N₂O emission tiers from biological nitrogen removal systems, but the approach is widely applicable. Machine learning on small, imbalanced, and heterogeneous datasets often faces critical challenges:
* **Data leakage**: When observations from the same source (e.g., a published study) are split across training and testing sets, hidden dependencies can inflate performance metrics.
* **Overfitting**: Significant distribution shifts between training and testing sets—especially in features or target variables—can lead to overly optimistic validation outcomes.
* **Biased Evaluations**: With high heterogeneity and class imbalance, results may vary drastically depending on how the data is split. Without controlling for randomness, model evaluations become unreliable, e.g. splits generating better performance is reported while new data generating different splits causes significantly lower result.

These challenges are common in domains where data is limited and sourced from diverse origins, such as literature-based datasets or multi-center studies. To address them, this pipeline introduces a systematic strategy for **data splitting, randomness assessment, and data leakage management**, enabling more reliable, fair, and reproducible model development.

2. Work Flow
- Grouped data split to avoid data leakage.
    A function is defined to conduct data splitting with:

    * Group Size Calculation: First computes how many rows each source contributes.
    * Proportional Accumulation: Randomly adds whole sources to test set until reaching ≈ `test_size` % of total rows. Default `test_size` = 20.
    * The test set has similar label distribution (based on `target_col`) as the overall dataset. The test set size is within a defined range (±`size_tolerance` of the desired `test_size`). Defaut `size_tolerance` = 0.05, `divergence_threshold` = 0.02.
    * Multiple randomized attempts (`max_attempts`, default = 50) are made to find the best-performing split using a divergence score and size penalty.
    * Result Guarantees: All rows from same source stay together. Handles any source size distribution.

Generates a new data splitting method ensuring: 1) Data from the same source are exclusively used as either training or testing set. 2) Training and Testing sets share similar y label distribution.

- Initial seed screening: identify seeds ensuring y distribution similarity in the testing set

    * Data splittings satisfying both the above requirements can actually be scarce depending on the data heterogeneity, hence a pre-screening is performed to select the seeds corresponding to valid data splittings.
    * `n_test` consecutive seeds (default `n_test` = 2000) were tried and analyzed for their group consistency, y label distribution and test size. 
    * Output: 
        `full_split_records.csv` - full records of `test_size`, `test_ratio`, `test_groups` (sources), and `label_distribution` in the test set for each random state.

        `unique_splits.csv` - unique splits among the tested random states, with the `test_groups` and its `group_count`, `occurences` and corresponding `random_states`.

        `split_analysis.png` - visualization of group assignment heatmap, consistency distribution, label distribution stability and test sizes distribution.
    
Generates certain amount of seeds ensuring grouped splitting and y distribution, and a series of visualization of seeds analysis. If data heterogeneity is relatively low for the X features, seeds listed in `unique_splits.csv` may be sufficient to avoid overfitting in following modeling steps.

- Refined seed screening: identify seeds ensuring similar X distribution from seeds identified last step
        
    * Due to the huge neterogeneity of data, X distribution might still be very different in the testing set which potentially casuses overfitting.
    * When there are multiple features, ensuring similar distribution in the testing set for every one of them can be extremely difficult and computationally expensive. Hence, we need to further screen the seeds ensuring similar y distribution for further screening.
    * Kolmogorov-Smirno Distance (`ks_stat`), Wasserstein Distance (`wasser_dist`) and Normalized Wasserstein (`norm_wasser`) are calculated for the numeric X features between the testing set and those in the training set.
    * Total Variation Distance (`TVD`) and Jensen-Shannon Divergence (`js_div`) are calculated for the categorical X features between the testing set and those in the training set.
    * Then, `wasser_dist` (default max acceptable threshold: 0.05) and `js_div` (default max acceptable threshold: 0.08) are used as the measure of distribution similarity for numeric and categorical variables, respectively, to select the good seeds holding `min_pass_rate` (default = 90%) of the features.
    * Output: 
    `distribution_differences.csv` - Divergence metrics for all features (`ks_stat`, `wasser_dist` and `norm_wasser` for numeric; `TVD` and `js_div` for categorical) at each random state.
    
    `distribution_summary.csv` - Mean values and standard diviations of the divergence metrics for all features (`ks_stat`, `wasser_dist` and `norm_wasser` for numeric; `TVD` and `js_div` for categorical) across all random states.

    `seed_summary.csv` - Mean values and standard diviations of the divergence metrics across all features (`ks_stat`, `wasser_dist` and `norm_wasser` for numeric; `TVD` and `js_div` for categorical) at each random state.

    `seed_quality.csv` - Summary at each random state for the number of features, features passing the divergence threshold, with a column `good_seed` specifying whether the split holding > 90% feature passing rate.

## Why Use This Tool
| Scenario | Why It Matters | How This Tool Helps |
| ---------| -------------- | --------------- |
| ✅ Heterogeneous sources (e.g., from literature, sites, or batches) | Risk of data leakage among groups   | Splits data at group level to prevent information leakage    |
| ✅ Small and imbalanced datasets | Standard splitting may overfit   | Ensures robust splits by fixing seeds and ranking divergence |
| ✅ Distribution drift between train and test sets | Can lead to over-optimistic test results      | Ranks and selects splits with minimal divergence             |
| 🚫 Homogeneous, large datasets  | Simpler splitting may suffice  | This tool may be overkill; consider `StratifiedKFold`        |
| 🚫 Time-series data   | This tool is not designed for online settings | Use time-aware or sequential splits  |

> For other splitting methods, consult:

> Zhu, J.-J., Yang, M., & Ren, Z. J. (2023). Machine Learning in Environmental Research: Common Pitfalls and Best Practices. *Environmental Science & Technology*, 57(46), 17671–17689. [https://doi.org/10.1021/acs.est.3c00026](https://doi.org/10.1021/acs.est.3c00026).

## Repo Structure
```bash
📁 FairSplit/
│
├── 📂 example/
│   ├── example_dataset.csv      # Example input file (grouped tabular data)
│   └── example.py               # Example code
│
├── 📂 src/
│   ├── split_function.py       # Core functions for group-level splitting while ensuring y distribution
│   ├── initial_screening.py    # Functions for identifying unique splits ensuring y similarity & visualization
│   └── refined_screeing.py     # Functions for summarizing the splits of X distribution similarity and identifying good seeds
│
├── README.md                   # This file
└── requirement.txt            # Dependencies
```

## How to Use
1. Installation
```bash
git clone https://github.com/YuqingYan0523/FairSplit.git
cd FairSplit
pip install -r requirement.txt
```
2. Example Use Case
```python
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
    FR_CSV = os.path.join(CURRENT_DIR, 'full_split_records.csv')
    UQ_CSV = os.path.join(CURRENT_DIR, 'unique_splits.csv')
    # Run analysis
    full_results, unique_splits_df = run_split_analysis(
        df,
        group_col='Source',
        target_col='y_var',
        n_tests=2000,
        test_size=0.2,
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
    max_attempts=50,
    size_tolerance=0.05,
    divergence_threshold=0.02
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
```
3. Divergence Metrics Selection

Current version uses:
* Mean values' difference for numerical y vairables
* Sum of the differences at each level of categorical y variables
to measure the y distribution divergence in the splitter function.

While Wasserstein Distance and Jensen-Shannon Divergence are used as the primary measure of X feature's distribution for numerical and categorical features, respectively. The code can be modified to change other divergence metrics depending on different feature's characteristics and model requirements.

4. Output and Follow-ups
This pipeline generates a list of seeds ensuring groupd data splitting and similarity of y variables in the testing set compared to the whole dataset, with per-seed evaluation of X features' divergence between the testing and training sets. Depending on the risk of overfitting, data splits fixed by seeds can be selected to conduct robust and reproducible modeling.
