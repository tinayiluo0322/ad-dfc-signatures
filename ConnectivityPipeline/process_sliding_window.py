"""
Sliding Window Connectivity Matrix Processor

This script processes brain region time series data using a sliding window approach
to generate dynamic functional connectivity matrices. The sliding window technique
captures temporal variations in brain connectivity patterns.

Key parameters:
- Window size: Number of time points in each window (default: 5, corresponding to 15 seconds)
- Step size: Number of time points to advance between windows (default: 2, corresponding to 10 seconds)

The script processes multiple subjects and generates connectivity matrices for each window,
which can then be used for downstream analysis of dynamic functional connectivity.
"""

import glob
import multiprocessing
import sys
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import networkx as nx
import math
import json
from typing import List, Tuple, Optional, Dict, Any


# Global list to track files with processing issues
files_with_issues: List[str] = []


def load_time_series_data(
    dataset_name: str, data_folder_path: str, label_file_path: Optional[str] = None
) -> Tuple[List[str], List[np.ndarray], List[Tuple]]:
    """
    Load time series data files and associated subject information.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset being processed
    data_folder_path : str
        Path to folder containing .npy time series files
    label_file_path : str, optional
        Path to JSON file containing subject labels

    Returns:
    --------
    tuple
        - List of file paths
        - List of time series arrays
        - List of subject information tuples (subject_id, session, label, filename, new_filename, group)
    """
    time_series_files = glob.glob(os.path.join(data_folder_path, "*.npy"))
    subject_info_list = []
    label_dict = {}
    time_series_data = []

    # Load subject labels if provided
    if label_file_path and os.path.exists(label_file_path):
        with open(label_file_path, "r") as f:
            label_dict = json.load(f)
        print(f"Loaded labels: {set(label_dict.values())}")
    else:
        print("No label dictionary provided.")

    # Process each time series file
    for ts_file_path in time_series_files:
        basename = os.path.basename(ts_file_path)

        # Extract subject ID and session information
        subject_id = basename.split("_")[0].split("-")[1]
        label = label_dict.get(basename.split("_")[0], None)
        session = (
            int(basename.split("_")[1].split("-")[1]) if dataset_name != "ADRC" else -1
        )

        # Create new filename for output
        new_filename = basename.replace("time_series", "connectivity_matrices")

        subject_info_list.append(
            (subject_id, session, label, basename, new_filename, -1)
        )

        # Load time series data
        time_series = np.load(ts_file_path)
        time_series_data.append(time_series)

    return time_series_files, time_series_data, subject_info_list


def group_data_by_length(
    dataset_name: str,
    time_series_files: List[str],
    subject_info_list: List[Tuple],
    time_series_data: List[np.ndarray],
) -> Tuple[List[List], List[List], List[Tuple], List[List]]:
    """
    Group time series data based on length criteria specific to each dataset.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset ('OASIS')
    time_series_files : list
        List of file paths
    subject_info_list : list
        List of subject information tuples
    time_series_data : list
        List of time series arrays

    Returns:
    --------
    tuple
        - Grouped file paths
        - Grouped time series data
        - Updated subject info list
        - Grouped subject info lists
    """
    grouped_files: List[List] = []
    grouped_data: List[List] = []
    grouped_subject_info: List[List] = []

    if dataset_name == "OASIS":
        # Group into 5 length-based categories
        length_ranges = [(90, 118), (136, 159), (160, 164), (292, 300), (599, 600)]
        grouped_data = [[] for _ in range(len(length_ranges))]
        grouped_files = [[] for _ in range(len(length_ranges))]
        grouped_subject_info = [[] for _ in range(len(length_ranges))]

        for i in reversed(range(len(time_series_data))):
            ts_data = time_series_data[i]
            file_path = time_series_files[i]
            subj_info = subject_info_list[i]
            length = ts_data.shape[1]

            # Find matching length range
            matched_groups = []
            for j, (start, end) in enumerate(length_ranges):
                if start <= length <= end:
                    matched_groups.append(j)

            if len(matched_groups) > 1:
                print(
                    f"Warning: File {file_path} fits multiple ranges {matched_groups}"
                )
            elif len(matched_groups) == 1:
                group_idx = matched_groups[0]
                grouped_data[group_idx].append(ts_data)
                grouped_files[group_idx].append(file_path)

                # Update subject info with group assignment
                subj_info_list = list(subj_info)
                subj_info_list[5] = group_idx
                grouped_subject_info[group_idx].append(tuple(subj_info_list))
            else:
                print(f"Info: File {file_path} does not fit any length range.")
                # Remove from original lists
                time_series_data.pop(i)
                subject_info_list.pop(i)
                time_series_files.pop(i)

        # Print group statistics
        for i, group in enumerate(grouped_data):
            print(
                f"Group {i+1} (length {length_ranges[i][0]}-{length_ranges[i][1]}): {len(group)} subjects"
            )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return grouped_files, grouped_data, subject_info_list, grouped_subject_info


def compute_length_statistics(time_series_data: List[np.ndarray]) -> Tuple[int, int]:
    """
    Compute and display statistics about time series lengths.

    Parameters:
    -----------
    time_series_data : list
        List of time series arrays

    Returns:
    --------
    tuple
        Maximum and minimum lengths
    """
    lengths = [ts.shape[1] for ts in time_series_data]
    max_length = max(lengths)
    min_length = min(lengths)
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)

    print("Time Series Length Statistics:")
    print(
        f"Max: {max_length}, Min: {min_length}, Mean: {mean_length:.2f}, Std: {std_length:.2f}\n"
    )

    return max_length, min_length


def compute_window_connectivity_matrix(
    window_data: np.ndarray, file_path: Optional[str] = None
) -> np.ndarray:
    """
    Compute connectivity matrix for a single window using Pearson correlation.

    Parameters:
    -----------
    window_data : np.ndarray
        Time series data for the window (regions x time_points)
    file_path : str, optional
        File path for error reporting

    Returns:
    --------
    np.ndarray
        Connectivity matrix (correlation matrix)
    """
    try:
        # Compute Pearson correlation matrix
        connectivity_matrix = np.corrcoef(window_data)

        # Handle constant time series (zero standard deviation)
        std_devs = np.std(window_data, axis=1)
        zero_std_mask = std_devs == 0

        if np.any(zero_std_mask):
            print(
                f"Warning: {np.sum(zero_std_mask)} regions have constant values in window"
            )
            if file_path:
                files_with_issues.append(file_path)
            # Set correlations involving constant regions to zero
            connectivity_matrix[zero_std_mask, :] = 0
            connectivity_matrix[:, zero_std_mask] = 0

        return connectivity_matrix

    except Exception as e:
        print(f"Error computing connectivity matrix for {file_path}: {e}")
        if file_path:
            files_with_issues.append(file_path)
        # Return identity matrix as fallback
        return np.eye(window_data.shape[0])


def save_connectivity_matrices(
    connectivity_matrices: List[np.ndarray],
    window_size: int,
    step_size: int,
    subject_info: Tuple,
    dataset_name: str = "unknown",
) -> None:
    """
    Save connectivity matrices to file.

    Parameters:
    -----------
    connectivity_matrices : list
        List of connectivity matrices
    window_size : int
        Size of sliding window
    step_size : int
        Step size between windows
    subject_info : tuple
        Subject information tuple
    dataset_name : str
        Name of the dataset
    """
    # Create output directory structure.  We place each subject in its own
    # folder that ends with "time_series" so that downstream distance
    # extraction scripts can locate the data consistently.
    label = subject_info[2] if subject_info[2] else "UNKNOWN"

    # Derive a folder name consistent with the original time-series naming
    # convention, e.g.  "sub-XXX_ses-YYY_time_series" (no .npy extension).
    original_basename: str = subject_info[3]  # e.g. sub-XXX_ses-YYY_time_series.npy
    subject_folder = original_basename.replace(".npy", "")

    directory = (
        f"connectivity_matrices/{dataset_name}/window_{window_size}_step_{step_size}/"
        f"{label}/{subject_folder}"
    )
    os.makedirs(directory, exist_ok=True)

    # Save each window connectivity matrix as an individual 2-D .npy file so
    # that they can be iterated over easily.  This avoids loading a large 3-D
    # array all at once and matches the expectations of the distance
    # extraction code which loops over *.npy files within each folder.
    for idx, matrix in enumerate(connectivity_matrices):
        window_fname = f"win_{idx:04d}.npy"
        output_path = os.path.join(directory, window_fname)
        np.save(output_path, matrix)

    if not os.listdir(directory):
        print(f"Warning: No files were created in {directory}")


def process_single_subject(
    file_path: str,
    window_size: int,
    step_size: int,
    min_length: int,
    subject_info: Tuple,
    dataset_name: str = "unknown",
) -> List[np.ndarray]:
    """
    Process a single subject's time series data with sliding window approach.

    Parameters:
    -----------
    file_path : str
        Path to the time series file
    window_size : int
        Size of sliding window
    step_size : int
        Step size between windows
    min_length : int
        Minimum length to clip time series to
    subject_info : tuple
        Subject information tuple
    dataset_name : str
        Name of the dataset

    Returns:
    --------
    list
        List of connectivity matrices for each window
    """
    # Load time series data
    time_series = np.load(file_path)

    # Clip to minimum length if necessary
    if time_series.shape[1] > min_length:
        time_series = time_series[:, :min_length]

    connectivity_matrices = []

    # Apply sliding window
    num_windows = (time_series.shape[1] - window_size) // step_size + 1

    for window_idx in range(num_windows):
        start_idx = window_idx * step_size
        end_idx = start_idx + window_size

        # Extract window data
        window_data = time_series[:, start_idx:end_idx]

        # Compute connectivity matrix for this window
        connectivity_matrix = compute_window_connectivity_matrix(window_data, file_path)
        connectivity_matrices.append(connectivity_matrix)

    print(
        f"Processed {len(connectivity_matrices)} windows for subject {subject_info[0]}"
    )

    return connectivity_matrices


def process_subject_wrapper(args: Tuple) -> Tuple[str, bool]:
    """
    Wrapper function for multiprocessing of individual subjects.

    Parameters:
    -----------
    args : tuple
        Arguments tuple containing (index, file_path, subject_info_list, additional_params)

    Returns:
    --------
    tuple
        File path and success status
    """
    local_idx, file_path, subject_info_list = args
    subject_info = subject_info_list[local_idx]

    # Get processing parameters from global variables or defaults
    window_size = 5  # Default window size
    step_size = 2  # Default step size
    min_length = 90  # Default minimum length
    dataset_name = "OASIS"

    try:
        # Process the subject
        connectivity_matrices = process_single_subject(
            file_path, window_size, step_size, min_length, subject_info, dataset_name
        )

        # Save results
        save_connectivity_matrices(
            connectivity_matrices, window_size, step_size, subject_info, dataset_name
        )

        return (file_path, False)  # Success

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return (file_path, True)  # Failure


def main():
    """
    Main processing function.
    """
    # Default parameters
    dataset_name = "OASIS"
    window_size = 5  # Corresponds to 15 seconds with 3-second TR
    step_size = 2  # Corresponds to 10-second steps

    # Parse command line arguments
    if len(sys.argv) >= 2:
        dataset_name = sys.argv[1]
    if len(sys.argv) >= 3:
        window_size = int(sys.argv[2])
    if len(sys.argv) >= 4:
        step_size = int(sys.argv[3])

    print(f"Processing dataset: {dataset_name}")
    print(f"Window size: {window_size} time points")
    print(f"Step size: {step_size} time points")

    # Set data paths (modify these paths as needed)
    data_folder_path = f"data/{dataset_name}/time_series"
    label_file_path = (
        f"data/{dataset_name}/labels.json" if dataset_name != "OASIS" else None
    )

    # Load data
    print("Loading time series data...")
    time_series_files, time_series_data, subject_info_list = load_time_series_data(
        dataset_name, data_folder_path, label_file_path
    )

    print(f"Loaded {len(time_series_files)} subjects")

    # Group data by length
    print("Grouping data by length...")
    grouped_files, grouped_data, subject_info_list, grouped_subject_info = (
        group_data_by_length(
            dataset_name, time_series_files, subject_info_list, time_series_data
        )
    )

    # Process each group
    for group_idx, (group_files, group_subject_info) in enumerate(
        zip(grouped_files, grouped_subject_info)
    ):
        if not group_files:
            continue

        group_name = f"Group_{group_idx + 1}"
        print(f"\n{'='*50}")
        print(f"Processing {group_name}")
        print(f"{'='*50}")

        # Compute statistics for this group
        group_data = grouped_data[group_idx]
        max_length, min_length = compute_length_statistics(group_data)

        # Estimate number of windows
        num_windows = (min_length - window_size) // step_size + 1
        print(f"Estimated number of windows per subject: {num_windows}")

        # Prepare arguments for multiprocessing
        args_list = [
            (local_idx, file_path, group_subject_info)
            for local_idx, file_path in enumerate(group_files)
        ]

        # Process subjects in parallel
        num_processes = multiprocessing.cpu_count()
        print(f"Using {num_processes} CPU cores for parallel processing")

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_subject_wrapper, args_list)

        # Report results
        failed_files = [file_path for file_path, failed in results if failed]
        if failed_files:
            print(f"Failed to process {len(failed_files)} files:")
            for file_path in failed_files:
                print(f"  - {file_path}")
        else:
            print(
                f"Successfully processed all {len(group_files)} subjects in {group_name}"
            )

    # Report files with issues
    if files_with_issues:
        print(f"\nFiles with processing issues: {len(files_with_issues)}")
        for file_path in files_with_issues:
            print(f"  - {file_path}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
