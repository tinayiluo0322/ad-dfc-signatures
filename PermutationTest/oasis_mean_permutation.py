# -*- coding: utf-8 -*-
"""
Title: Permutation-Based Testing of Mean-Based dFC Signatures in Alzheimer’s Disease

Key Functions:
- Loads time series distance metrics across diagnostic groups (Normal, MCI, Dementia)
- Computes mean summary features per subject from distance time series
- Regresses out age effects from individual-level features
- Conducts two-tailed permutation tests to evaluate group differences
- Outputs group-level statistics and p-values across multiple distance metrics:
  - Spectral
  - Nuclear
  - Manhattan
  - Chebyshev
  - Frobenius
  - 0-homology Wasserstein
  - 1-homology Wasserstein

Note:
This script performs five independent trials per dataset using different random seeds, 
with 50,000 permutations each, to ensure statistical robustness and reproducibility.
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import trapz
import time
from tqdm import tqdm
from google.colab import drive
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.lines as mlines
import seaborn as sns
import os

# Mount Google Drive
drive.mount('/content/drive')

# Set base seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def parse_filename(filename):
    """Extract metadata from filename."""
    parts = filename.split('_')
    if len(parts) < 8:
      raise ValueError(f"Unexpected filename format: {filename}")
    dataset = parts[0]  # Oasis 
    data_type = parts[3]  # Metrics Type
    comparison = f"{parts[4]}_{parts[5]}"  # Normal vs Dementia or Normal vs MCI
    sex_data = parts[6].split('.')[0]  # Male or Female
    return dataset, data_type, comparison, sex_data

def load_dataset(file_path):
    """Load CSV dataset and extract relevant columns."""
    df = pd.read_csv(file_path)
    required_columns = {'subject','label', 'sex', 'group', 'age'}
    if not required_columns.issubset(df.columns):
      raise ValueError(f"Missing required columns: {required_columns - set(df.columns)} in {file_path}")
    groups = df['label'].copy()
    age = df['age'].copy()
    print(f"Unique group labels in dataset: {groups.unique()}")  # Extract group labels (Normal, Dementia, MCI)
    print(f"Unique sex in the dataset: {df['sex'].unique()}")  # Extract unique sex values
    print(f"Unique group in the dataset: {df['group'].unique()}")  # Extract unique group values
    time_series_data = df.drop(columns=['subject', 'label', 'sex', 'group', 'age'])
    return df, time_series_data, groups, age

def compute_mean_per_timestep(time_series_data):
    """Compute the mean value across all available time steps per individual while handling NaN values."""

    # Compute the mean across non-NaN time steps for each row. print out the length of the 5 groups of timestamps for double check
    filtered_lengths = np.array([np.count_nonzero(~np.isnan(row)) for row in time_series_data.values])
    unique_lengths, counts = np.unique(filtered_lengths, return_counts=True)
    print("Unique timestep lengths after NaN filtering:", dict(zip(unique_lengths, counts)))

    # Compute the mean for each row filtering out NaNs
    mean_values = np.nanmean(time_series_data.values, axis=1)

    if np.isnan(mean_values).any():
        print("Warning: NaN values detected in mean_values! Investigating...")
        print("Rows with NaN in mean_values:", np.where(np.isnan(mean_values)))

    return mean_values

def adjust_for_age(mean_df_original, dataset, data_type, comparison, sex_data):
    """
    Regress out age from means across the entire dataset and return adjusted values.
    Plots include regression lines with confidence bands.
    """
    # Extract age and means
    age = mean_df_original['Age'].values.reshape(-1, 1)
    means = mean_df_original['Mean_Value'].values

    # Fit linear regression model on the entire dataset
    model = LinearRegression()
    model.fit(age, means)
    predicted = model.predict(age)
    residuals = means - predicted  # Adjusted means (residuals)

    # Add residuals to the DataFrame
    mean_df_original['Adjusted_Mean'] = residuals

    # Set up colors for each group
    unique_groups = mean_df_original['Group'].unique()
    colors = {
        'Normal': '#1f77b4',    # Blue
        'Dementia': '#d62728',  # Red
        'MCI': '#ffd700'        # Yellow
    }

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Plot unadjusted values with groups
    sns.scatterplot(
        ax=axes[0],
        x=age.flatten(),
        y=means,
        hue=mean_df_original['Group'],
        palette=colors,
        alpha=0.6
    )

    # Add regression line with confidence band for unadjusted data using seaborn's regplot
    sns.regplot(
        x=age.flatten(),
        y=means,
        ax=axes[0],
        scatter=False,  # Don't add more scatter points
        color='black',  # Black regression line
        line_kws={'linestyle':'--', 'linewidth':2, 'label':'Regression Line'},
        ci=95  # 95% confidence interval
    )

    # Plot adjusted values (residuals) with groups
    sns.scatterplot(
        ax=axes[1],
        x=age.flatten(),
        y=residuals,
        hue=mean_df_original['Group'],
        palette=colors,
        alpha=0.6
    )

    # Add regression line with confidence band for adjusted data using seaborn's regplot
    sns.regplot(
        x=age.flatten(),
        y=residuals,
        ax=axes[1],
        scatter=False,  # Don't add more scatter points
        color='black',  # Black regression line
        line_kws={'linestyle':'--', 'linewidth':2, 'label':'Regression Line'},
        ci=95  # 95% confidence interval
    )

    # Add a horizontal line at y=0 for reference in residuals plot
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Customize first plot
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Mean")
    axes[0].set_title(f"{dataset} {data_type} {comparison} {sex_data}\n Age vs. Original Mean")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Fix the legend for first plot (add regression line manually)
    handles, labels = axes[0].get_legend_handles_labels()
    line_handle = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='Regression Line')
    handles.append(line_handle)
    axes[0].legend(handles=handles)

    # Customize second plot
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Adjusted Mean (Residuals)")
    axes[1].set_title(f"{dataset} {data_type} {comparison} {sex_data}\nAge vs. Adjusted Mean")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Fix the legend for second plot (add regression line manually)
    handles, labels = axes[1].get_legend_handles_labels()
    handles.append(line_handle)  # Reuse the same line handle
    axes[1].legend(handles=handles)

    plt.tight_layout()
    plt.show()

    return mean_df_original['Adjusted_Mean'].values

def permutation_test(mean_df, observed_statistic, dataset, data_type, comparison, sex_data, seed, num_permutations=50000):
    """Perform permutation test and return p-value with a given seed."""
    np.random.seed(seed)  # Set different seed per trial for deterministic results
    perm_stats = np.zeros(num_permutations)

    print("Starting two-tailed permutation test...")

    for i in tqdm(range(num_permutations), desc="Running Permutations", ncols=80):
        perm_df = mean_df.copy()
        perm_df['Shuffled_Group'] = np.random.permutation(perm_df['Group'].values)  # Shuffle only group labels

        condition_group = [g for g in perm_df['Shuffled_Group'].unique() if g != 'Normal'][0]
        perm_mean_condition = perm_df[perm_df['Shuffled_Group'] == condition_group]['Adjusted_Mean'].mean()
        perm_mean_normal = perm_df[perm_df['Shuffled_Group'] == 'Normal']['Adjusted_Mean'].mean()

        perm_stats[i] = abs(perm_mean_condition - perm_mean_normal)

    p_value = np.mean(perm_stats >= observed_statistic)
    print(f"P-value for seed {seed}: {p_value}")

    # Visualization of permutation test results
    plt.figure(figsize=(8, 5))
    plt.hist(perm_stats, bins=50, edgecolor='black', alpha=0.7, label="Permutation Distribution")
    plt.axvline(observed_statistic, color='red', linestyle='dashed', linewidth=2, label="Observed Statistic")
    plt.xlabel("Permutation Test Statistic")
    plt.ylabel("Frequency")
    plt.title(f"Two Tailed Permutation Test Distribution {dataset} {data_type} {comparison} {sex_data}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    return p_value

def run_pipeline(file_paths, num_permutations=50000, num_trials=5):
    """Run permutation tests on multiple datasets and store results."""
    results = []
    trial_seeds = [SEED + i for i in range(num_trials)]  # Generate 5 different seeds

    for file_path in file_paths:
        print("-" * 100)
        print(f"Processing: {file_path}")
        dataset, data_type, comparison, sex_data = parse_filename(os.path.basename(file_path))
        print(dataset, data_type, comparison, sex_data)
        df, time_series_data, groups, age = load_dataset(file_path)
        mean_values = compute_mean_per_timestep(time_series_data)

        # # Create DataFrame storing computed values with group labels
        # mean_df = pd.DataFrame({'Mean_Value': mean_values, 'Group': groups.values})
        # print("Sample computed mean data:")
        # print(mean_df.head())  # Display first few rows for verification

        # Create DataFrame storing computed values with group labels
        mean_df_original = pd.DataFrame({
            'Mean_Value': mean_values,
            'Group': groups.values,
            'Age': age.values
        }, index=groups.index)

        print("Sample original data:")
        print(mean_df_original.head())  # Display first few rows for verification

        # Adjust for age
        adjusted_means = adjust_for_age(mean_df_original, dataset, data_type, comparison, sex_data)

        # Create DataFrame storing computed values with group labels
        mean_df = pd.DataFrame({
            'Adjusted_Mean': adjusted_means,
            'Group': groups.values
        }, index=groups.index)

        print("Sample adjusted data:")
        print(mean_df.head())  # Display first few rows for verification

        # Compute observed statistic once per dataset
        condition_group = [g for g in groups.unique() if g != 'Normal'][0]  # Dynamically detect condition group (Dementia or MCI)
        mean_condition = mean_df[mean_df['Group'] == condition_group]['Adjusted_Mean'].mean()
        mean_normal = mean_df[mean_df['Group'] == 'Normal']['Adjusted_Mean'].mean()
        observed_statistic = abs(mean_condition - mean_normal)
        print(f"Observed Statistic: {observed_statistic}")

        # Perform 5 trials with different seeds
        trial_p_values = [permutation_test(mean_df, observed_statistic, dataset, data_type, comparison, sex_data, trial_seeds[i], num_permutations) for i in range(num_trials)]
        print(f"P values throughout 5 trials: {trial_p_values}")
        avg_p = np.mean(trial_p_values)
        std_p = np.std(trial_p_values)

        results.append([dataset, data_type, comparison, sex_data, observed_statistic, trial_p_values, avg_p, std_p])
        # print out the results
        print("-" * 50)
        print(f"Processed Dataset: {dataset}, Data Type: {data_type}, Comparison: {comparison}, Sex: {sex_data}")
        print(f"Observed Statistic: {observed_statistic:.4f}")
        print(f"P-values (5 Trials): {trial_p_values}")
        print(f"Average P-Value: {avg_p:.5f}, Standard Deviation: {std_p:.5f}")

    results_df = pd.DataFrame(results, columns=['Dataset', 'Data Type', 'Comparison', 'Sex', 'Observed Statistic', 'P-Values (5 Trials)', 'Avg P-Value', 'Std Dev'])
    return results_df

# Spectral
file_names = [
    "Oasis_all_data_spectral_normal_dementia_male_age.csv",
    "Oasis_all_data_spectral_normal_dementia_female_age.csv",
    "Oasis_all_data_spectral_normal_mci_female_age.csv",
    "Oasis_all_data_spectral_normal_mci_male_age.csv",
]
file_paths = [f"/content/drive/My Drive/wass_ts/Oasis/Permutation Test/full_data_spectral/{f}" for f in file_names]

#Run pipeline and save results
results_df = run_pipeline(file_paths)
output_path = "/content/drive/My Drive/wass_ts/Oasis/Permutation Test/full_data_spectral_permutation_results/Oasis_spectral_full_data_mean_regress_age_permutation_results.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results_df.to_csv(output_path, index=False)

#Nuclear Norm
file_names = [
    "Oasis_all_data_nuclear_normal_dementia_male_age.csv",
    "Oasis_all_data_nuclear_normal_dementia_female_age.csv",
    "Oasis_all_data_nuclear_normal_mci_female_age.csv",
    "Oasis_all_data_nuclear_normal_mci_male_age.csv",
]
file_paths = [f"/content/drive/My Drive/wass_ts/Oasis/Permutation Test/full_data_nuclear/{f}" for f in file_names]

#Run pipeline and save results
results_df = run_pipeline(file_paths)
output_path = "/content/drive/My Drive/wass_ts/Oasis/Permutation Test/full_data_nuclear_permutation_results/Oasis_nuclear_full_data_mean_regress_age_permutation_results.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results_df.to_csv(output_path, index=False)

# Manhattan
file_names = [
    "Oasis_all_data_manhattan_normal_dementia_male_age.csv",
    "Oasis_all_data_manhattan_normal_dementia_female_age.csv",
    "Oasis_all_data_manhattan_normal_mci_female_age.csv",
    "Oasis_all_data_manhattan_normal_mci_male_age.csv",
]
file_paths = [f"/content/drive/My Drive/wass_ts/Oasis/Permutation Test/full_data_manhattan/{f}" for f in file_names]

#Run pipeline and save results
results_df = run_pipeline(file_paths)
output_path = "/content/drive/My Drive/wass_ts/Oasis/Permutation Test/full_data_manhattan_permutation_results/Oasis_manhattan_full_data_mean_regress_age_permutation_results.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results_df.to_csv(output_path, index=False)

# Chebyshev
file_names = [
    "Oasis_all_data_chebyshev_normal_dementia_male_age.csv",
    "Oasis_all_data_chebyshev_normal_dementia_female_age.csv",
    "Oasis_all_data_chebyshev_normal_mci_female_age.csv",
    "Oasis_all_data_chebyshev_normal_mci_male_age.csv",
]
file_paths = [f"/content/drive/My Drive/wass_ts/Oasis/Permutation Test/full_data_chebyshev/{f}" for f in file_names]

#Run pipeline and save results
results_df = run_pipeline(file_paths)
output_path = "/content/drive/My Drive/wass_ts/Oasis/Permutation Test/full_data_chebyshev_permutation_results/Oasis_chebyshev_full_data_mean_regress_age_permutation_results.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results_df.to_csv(output_path, index=False)

# Frobenius
file_names = [
    "Oasis_all_data_frobenius_normal_dementia_male_age.csv",
    "Oasis_all_data_frobenius_normal_dementia_female_age.csv",
    "Oasis_all_data_frobenius_normal_mci_female_age.csv",
    "Oasis_all_data_frobenius_normal_mci_male_age.csv",
]
file_paths = [f"/content/drive/My Drive/wass_ts/Oasis/Permutation Test/full_data_frobenius/{f}" for f in file_names]

#Run pipeline and save results
results_df = run_pipeline(file_paths)
output_path = "/content/drive/My Drive/wass_ts/Oasis/Permutation Test/full_data_frobenius_permutation_results/Oasis_frobenius_full_data_mean_regress_age_permutation_results.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results_df.to_csv(output_path, index=False)

# 0-homology 1 homology Wasserstein
file_names = [
    "Oasis_all_data_birth_normal_dementia_male_age.csv",
    "Oasis_all_data_birth_normal_dementia_female_age.csv",
    "Oasis_all_data_birth_normal_mci_female_age.csv",
    "Oasis_all_data_birth_normal_mci_male_age.csv",
    "Oasis_all_data_death_normal_dementia_male_age.csv",
    "Oasis_all_data_death_normal_dementia_female_age.csv",
    "Oasis_all_data_death_normal_mci_female_age.csv",
    "Oasis_all_data_death_normal_mci_male_age.csv"
]
file_paths = [f"/content/drive/My Drive/wass_ts/Oasis/Permutation Test/full_data/{f}" for f in file_names]

#Run pipeline and save results
results_df = run_pipeline(file_paths)
output_path = "/content/drive/My Drive/wass_ts/Oasis/Permutation Test/full_data_permutation_results/Oasis_full_data_mean_regress_age_permutation_results.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results_df.to_csv(output_path, index=False)
