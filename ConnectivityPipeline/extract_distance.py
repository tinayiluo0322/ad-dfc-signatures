"""
Connectivity Distance Analyzer

This script analyzes dynamic functional connectivity by computing distances between
consecutive connectivity matrices generated from sliding window analysis. Multiple
distance metrics are supported to capture different aspects of connectivity changes.

Supported distance metrics:
- Wasserstein distance (birth/death): Based on topological persistence features
- Frobenius distance: Matrix norm-based distance
- Spectral distance: Based on eigenvalue differences
- Nuclear distance: Based on singular value differences
- Manhattan distance: Element-wise L1 distance
- Chebyshev distance: Element-wise L∞ distance

The script processes connectivity matrices from multiple subjects and generates
time series of distances that can be used for further analysis.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import sys
import networkx as nx
import math
import multiprocessing
from functools import partial
from typing import List, Tuple, Optional, Dict, Any, Union


def plot_distance_time_series(
    distance_data: Union[Dict, pd.DataFrame],
    distance_type: str = "wasserstein",
    window_size: Optional[int] = None,
    step_size: Optional[int] = None,
    subject_info: Optional[Tuple] = None,
    notes: Optional[str] = None,
) -> List[float]:
    """
    Plot distance time series showing temporal changes in connectivity.

    Parameters:
    -----------
    distance_data : dict or pd.DataFrame
        Distance values, either as dictionary or DataFrame with 'distances' column
    distance_type : str
        Type of distance metric used
    window_size : int, optional
        Size of sliding window used
    step_size : int, optional
        Step size between windows
    subject_info : tuple, optional
        Subject information tuple
    notes : str, optional
        Additional notes for the plot

    Returns:
    --------
    list
        List of distance values
    """
    # Extract distance values from input data
    if isinstance(distance_data, pd.DataFrame):
        if "distances" in distance_data.columns:
            # Use concatenated distances from all subjects
            all_distances = np.concatenate(distance_data["distances"].values)
            # Convert to plain Python list to avoid type issues later
            distance_values = all_distances.tolist()

            # Extract metadata if not provided
            if subject_info is None and len(distance_data) > 0:
                row = distance_data.iloc[0]
                subject_info = (
                    row["label"],
                    row["subject"],
                    row["session"],
                    row["run"],
                    row["group"],
                )

            # Try to extract window parameters from notes
            if window_size is None and "notes" in distance_data.columns:
                notes_text = distance_data["notes"].iloc[0]
                if notes_text and "window" in notes_text and "step" in notes_text:
                    try:
                        window_part = notes_text.split("window_")[1].split("_")[0]
                        step_part = notes_text.split("step_")[1].split("_")[0]
                        window_size = int(window_part)
                        step_size = int(step_part)
                    except:
                        pass
        else:
            raise ValueError("DataFrame does not contain a 'distances' column")
    else:
        # Flatten dictionary values (each is a list of floats) into a single list
        distance_values = [
            float(v) for values in distance_data.values() for v in values
        ]

    print(f"Distance range: {min(distance_values):.4f} - {max(distance_values):.4f}")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Build plot title
    plot_title = f"{distance_type.replace('_', ' ').title()} Distance Time Series"
    if window_size is not None and step_size is not None:
        plot_title += f"\nWindow Size: {window_size}, Step Size: {step_size}"
    if subject_info:
        plot_title += f"\nSubject {subject_info[1]}, Session {subject_info[2]}"
    if notes:
        plot_title += f"\n{notes}"

    fig.suptitle(plot_title, fontsize=16)

    # Plot distance time series
    ax.plot(distance_values, color="dodgerblue", linewidth=1.5)
    ax.set_title(
        f"Temporal Changes in {distance_type.replace('_', ' ').title()} Distance"
    )
    ax.set_xlabel("Time Window")
    ax.set_ylabel("Distance")
    ax.grid(True, alpha=0.3)

    # Set axis limits with some padding
    ax.set_ylim(0, max(distance_values) * 1.1)
    ax.set_xlim(0, len(distance_values))

    plt.tight_layout()

    # Save plot
    output_dir = "distance_time_series_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename
    filename_parts = [distance_type, "time_series"]
    if window_size is not None and step_size is not None:
        filename_parts.append(f"window_{window_size}_step_{step_size}")
    if subject_info:
        filename_parts.append(f"sub_{subject_info[1]}_ses_{subject_info[2]}")
    if notes:
        filename_parts.append(notes.replace(" ", "_"))

    filename = "_".join(filename_parts) + ".png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    # Always return a list (not a NumPy array) to match the declared return type
    if isinstance(distance_values, np.ndarray):
        distance_values = distance_values.tolist()
    return distance_values


def adjacency_to_persistence(
    adjacency_matrix: np.ndarray,
) -> Tuple[List[float], List[float]]:
    """
    Convert adjacency matrix to persistence features (connected components and cycles).

    Parameters:
    -----------
    adjacency_matrix : np.ndarray
        Weighted adjacency matrix

    Returns:
    --------
    tuple
        - Connected component birth times
        - Cycle death times
    """
    # Create graph from adjacency matrix
    graph = nx.from_numpy_array(adjacency_matrix)

    # Compute maximum spanning tree for connected components
    mst = nx.maximum_spanning_tree(graph)
    mst_edges = mst.edges(data=True)

    # Extract connected component features (sorted edge weights in MST)
    connected_components = sorted(
        [edge[2]["weight"] for edge in mst_edges], reverse=True
    )

    # Remove MST edges to find cycles
    graph.remove_edges_from(mst_edges)
    remaining_edges = graph.edges(data=True)

    # Extract cycle features (sorted edge weights not in MST)
    cycles = sorted([edge[2]["weight"] for edge in remaining_edges], reverse=True)

    # Add zero-valued edges for complete cycle representation
    num_total_edges = (len(adjacency_matrix) * (len(adjacency_matrix) - 1)) // 2
    num_zero_edges = num_total_edges - len(connected_components) - len(cycles)
    if num_zero_edges > 0:
        cycles.extend([0] * int(num_zero_edges))

    return connected_components, cycles


def persistence_to_vector(
    connected_components: List[float],
    cycles: List[float],
    num_sampled_ccs: int,
    num_sampled_cycles: int,
) -> List[float]:
    """
    Convert persistence features to fixed-length feature vectors.

    Parameters:
    -----------
    connected_components : list
        Connected component birth times
    cycles : list
        Cycle death times
    num_sampled_ccs : int
        Number of connected components to sample
    num_sampled_cycles : int
        Number of cycles to sample

    Returns:
    --------
    list
        Combined feature vector
    """
    # Sample connected components uniformly
    num_ccs = len(connected_components)
    cc_indices = [
        math.ceil((i + 1) * num_ccs / num_sampled_ccs) - 1
        for i in range(num_sampled_ccs)
    ]

    # Sample cycles uniformly
    num_cycles = len(cycles)
    cycle_indices = [
        math.ceil((i + 1) * num_cycles / num_sampled_cycles) - 1
        for i in range(num_sampled_cycles)
    ]

    # Create feature vectors
    cc_array = np.array(connected_components)
    cycle_array = np.array(cycles)

    return list(cc_array[cc_indices]) + list(cycle_array[cycle_indices])


def calculate_wasserstein_distance(
    birth_vectors: List[List[float]], death_vectors: List[List[float]]
) -> Tuple[List[float], List[float]]:
    """
    Calculate Wasserstein distance between consecutive persistence vectors.

    Parameters:
    -----------
    birth_vectors : list
        List of birth feature vectors
    death_vectors : list
        List of death feature vectors

    Returns:
    --------
    tuple
        - Birth Wasserstein distances
        - Death Wasserstein distances
    """
    birth_distances = []
    death_distances = []

    for i in range(len(birth_vectors) - 1):
        # L2 distance between consecutive vectors
        birth_dist = np.linalg.norm(
            np.array(birth_vectors[i]) - np.array(birth_vectors[i + 1])
        )
        death_dist = np.linalg.norm(
            np.array(death_vectors[i]) - np.array(death_vectors[i + 1])
        )

        birth_distances.append(float(birth_dist))
        death_distances.append(float(death_dist))

    return birth_distances, death_distances


def calculate_frobenius_distance(
    connectivity_matrices: List[np.ndarray],
) -> List[float]:
    """
    Calculate Frobenius distance between consecutive connectivity matrices.

    Parameters:
    -----------
    connectivity_matrices : list
        List of connectivity matrices

    Returns:
    --------
    list
        Frobenius distances between consecutive matrices
    """
    distances = []
    for i in range(len(connectivity_matrices) - 1):
        distance = np.linalg.norm(
            connectivity_matrices[i] - connectivity_matrices[i + 1], "fro"
        )
        distances.append(float(distance))
    return distances


def calculate_spectral_distance(connectivity_matrices: List[np.ndarray]) -> List[float]:
    """
    Calculate spectral distance between consecutive connectivity matrices.

    Parameters:
    -----------
    connectivity_matrices : list
        List of connectivity matrices

    Returns:
    --------
    list
        Spectral distances between consecutive matrices
    """

    def spectral_distance(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
        """Compute spectral distance between two matrices."""
        eigenvals_a = np.linalg.eigvals(matrix_a)
        eigenvals_b = np.linalg.eigvals(matrix_b)

        # Sort eigenvalues for comparison
        eigenvals_a = np.sort(eigenvals_a)[::-1]
        eigenvals_b = np.sort(eigenvals_b)[::-1]

        return float(np.linalg.norm(eigenvals_a - eigenvals_b))

    distances = []
    for i in range(len(connectivity_matrices) - 1):
        distance = spectral_distance(
            connectivity_matrices[i], connectivity_matrices[i + 1]
        )
        distances.append(distance)

    return distances


def calculate_nuclear_distance(connectivity_matrices: List[np.ndarray]) -> List[float]:
    """
    Calculate nuclear (trace) distance between consecutive connectivity matrices.

    Parameters:
    -----------
    connectivity_matrices : list
        List of connectivity matrices

    Returns:
    --------
    list
        Nuclear distances between consecutive matrices
    """
    distances = []
    for i in range(len(connectivity_matrices) - 1):
        # Nuclear norm is the sum of singular values
        diff_matrix = connectivity_matrices[i] - connectivity_matrices[i + 1]
        distance = np.linalg.norm(diff_matrix, "nuc")
        distances.append(float(distance))

    return distances


def calculate_manhattan_distance(
    connectivity_matrices: List[np.ndarray],
) -> List[float]:
    """
    Calculate Manhattan (L1) distance between consecutive connectivity matrices.

    Parameters:
    -----------
    connectivity_matrices : list
        List of connectivity matrices

    Returns:
    --------
    list
        Manhattan distances between consecutive matrices
    """
    distances = []
    for i in range(len(connectivity_matrices) - 1):
        distance = np.sum(
            np.abs(connectivity_matrices[i] - connectivity_matrices[i + 1])
        )
        distances.append(float(distance))

    return distances


def calculate_chebyshev_distance(
    connectivity_matrices: List[np.ndarray],
) -> List[float]:
    """
    Calculate Chebyshev (L∞) distance between consecutive connectivity matrices.

    Parameters:
    -----------
    connectivity_matrices : list
        List of connectivity matrices

    Returns:
    --------
    list
        Chebyshev distances between consecutive matrices
    """
    distances = []
    for i in range(len(connectivity_matrices) - 1):
        distance = np.max(
            np.abs(connectivity_matrices[i] - connectivity_matrices[i + 1])
        )
        distances.append(float(distance))

    return distances


def extract_subject_info_from_path(folder_path: str) -> Tuple[str, str, int, int, str]:
    """
    Extract subject information from folder path.

    Parameters:
    -----------
    folder_path : str
        Path to subject's connectivity matrices folder

    Returns:
    --------
    tuple
        Subject information (label, subject_id, session, run, group)
    """
    path_parts = folder_path.split(os.sep)

    # Extract information from path structure
    # Assumes structure: .../label/subject_session_run_time_series/
    try:
        label = path_parts[-2] if path_parts[-1].endswith("_time_series") else "Unknown"
        folder_name = path_parts[-1].replace("_time_series", "")

        # Parse folder name for subject info
        name_parts = folder_name.split("_")
        subject_id = (
            name_parts[0].split("-")[1] if "-" in name_parts[0] else name_parts[0]
        )
        session = (
            int(name_parts[1].split("-")[1])
            if len(name_parts) > 1 and "-" in name_parts[1]
            else 1
        )
        run = (
            int(name_parts[2].split("-")[1])
            if len(name_parts) > 2 and "-" in name_parts[2]
            else 1
        )

        # Extract group information from path
        group = "Unknown"
        for part in path_parts:
            if part.startswith("Group_") or part.startswith("G"):
                group = part
                break

        return label, subject_id, session, run, group

    except (IndexError, ValueError) as e:
        print(f"Warning: Could not parse subject info from {folder_path}: {e}")
        return "Unknown", "Unknown", 1, 1, "Unknown"


def compute_wasserstein_distances(
    connectivity_matrices: List[np.ndarray],
    num_sampled_ccs: int,
    num_sampled_cycles: int,
) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
    """
    Compute Wasserstein distances from connectivity matrices using topological features.

    Parameters:
    -----------
    connectivity_matrices : list
        List of connectivity matrices
    num_sampled_ccs : int
        Number of connected components to sample
    num_sampled_cycles : int
        Number of cycles to sample

    Returns:
    --------
    tuple
        - Birth feature vectors
        - Death feature vectors
        - Birth Wasserstein distances
        - Death Wasserstein distances
    """
    birth_vectors = []
    death_vectors = []

    # Convert each connectivity matrix to persistence features
    for matrix in connectivity_matrices:
        connected_components, cycles = adjacency_to_persistence(matrix)
        feature_vector = persistence_to_vector(
            connected_components, cycles, num_sampled_ccs, num_sampled_cycles
        )

        # Split into birth and death components
        birth_vectors.append(feature_vector[:num_sampled_ccs])
        death_vectors.append(feature_vector[num_sampled_ccs:])

    # Calculate Wasserstein distances
    birth_distances, death_distances = calculate_wasserstein_distance(
        birth_vectors, death_vectors
    )

    return birth_vectors, death_vectors, birth_distances, death_distances


def process_subject_folder(
    folder_path: str, distance_type: str
) -> Optional[Tuple[Tuple, List[float]]]:
    """
    Process a single subject's connectivity matrices folder.

    Parameters:
    -----------
    folder_path : str
        Path to folder containing connectivity matrices
    distance_type : str
        Type of distance to compute

    Returns:
    --------
    tuple or None
        Subject info tuple and computed distances, or None if processing failed
    """
    try:
        # Extract subject information
        subject_info = extract_subject_info_from_path(folder_path)

        # Load connectivity matrices
        matrix_files = glob.glob(os.path.join(folder_path, "*.npy"))
        if not matrix_files:
            print(f"No .npy files found in {folder_path}")
            return None

        connectivity_matrices = []
        for matrix_file in sorted(matrix_files):
            try:
                matrix = np.load(matrix_file)
                connectivity_matrices.append(matrix)
            except Exception as e:
                print(f"Error loading {matrix_file}: {e}")

        if len(connectivity_matrices) < 2:
            print(
                f"Need at least 2 matrices for distance calculation, found {len(connectivity_matrices)}"
            )
            return None

        # Calculate distances based on type
        if distance_type.lower() in ["wasserstein_birth", "wasserstein_death"]:
            # Use default sampling parameters
            num_sampled_ccs = len(connectivity_matrices) // 2
            num_sampled_cycles = len(connectivity_matrices) - num_sampled_ccs

            _, _, birth_distances, death_distances = compute_wasserstein_distances(
                connectivity_matrices, num_sampled_ccs, num_sampled_cycles
            )

            distances = (
                birth_distances
                if distance_type.lower() == "wasserstein_birth"
                else death_distances
            )

        elif distance_type.lower() == "frobenius":
            distances = calculate_frobenius_distance(connectivity_matrices)
        elif distance_type.lower() == "spectral":
            distances = calculate_spectral_distance(connectivity_matrices)
        elif distance_type.lower() == "nuclear":
            distances = calculate_nuclear_distance(connectivity_matrices)
        elif distance_type.lower() == "manhattan":
            distances = calculate_manhattan_distance(connectivity_matrices)
        elif distance_type.lower() == "chebyshev":
            distances = calculate_chebyshev_distance(connectivity_matrices)
        else:
            print(f"Unknown distance type: {distance_type}")
            return None

        return (subject_info, distances)

    except Exception as e:
        print(f"Error processing {folder_path}: {e}")
        return None


def process_all_subjects(
    data_folder_path: str,
    distance_type: str = "wasserstein_birth",
    labels: Optional[List[str]] = None,
) -> Tuple[Dict[Tuple, List[float]], List[Tuple]]:
    """
    Process all subjects' connectivity matrices and compute distances.

    Parameters:
    -----------
    data_folder_path : str
        Root path containing subject folders
    distance_type : str
        Type of distance to compute
    labels : list, optional
        List of valid labels to include

    Returns:
    --------
    tuple
        - Dictionary mapping subject info to distances
        - List of subject information tuples
    """
    if labels is None:
        labels = ["Normal", "MCI", "Dementia", "Impaired-not-MCI", "Unknown"]

    print(f"Computing {distance_type} distances")
    print(f"Searching for connectivity matrix folders in {data_folder_path}")

    # Find all subject folders containing connectivity matrices
    subject_folders = glob.glob(
        os.path.join(data_folder_path, "**", "*_time_series"), recursive=True
    )
    print(f"Found {len(subject_folders)} subject folders")

    # Set up parallel processing
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")

    # Create partial function with distance type
    process_func = partial(process_subject_folder, distance_type=distance_type)

    # Process subjects in parallel
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(process_func, subject_folders)

    # Filter valid results and organize data
    distance_dict = {}
    subject_info_list = []

    for result in results:
        if result is not None:
            subject_info, distances = result
            if subject_info not in subject_info_list:
                subject_info_list.append(subject_info)
                distance_dict[subject_info] = distances
            else:
                print(f"Warning: Duplicate subject found: {subject_info}")

    print(f"Successfully processed {len(distance_dict)} subjects")
    return distance_dict, subject_info_list


def create_distance_dataframe(
    dataset_name: str,
    subject_info_list: List[Tuple],
    distance_dict: Dict[Tuple, List[float]],
    distance_type: str = "wasserstein_birth",
    notes: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create a pandas DataFrame from distance data.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    subject_info_list : list
        List of subject information tuples
    distance_dict : dict
        Dictionary mapping subject info to distances
    distance_type : str
        Type of distance computed
    notes : str, optional
        Additional notes

    Returns:
    --------
    pd.DataFrame
        DataFrame containing distance data
    """
    rows = []
    print(f"Creating DataFrame with {len(subject_info_list)} subjects")

    for subject_info in subject_info_list:
        label, subject_id, session, run, group = subject_info
        distances = distance_dict.get(subject_info, [])

        # Only include subjects with valid labels and distances
        if label != "Unknown" and distances:
            rows.append(
                {
                    "label": label,
                    "subject": subject_id,
                    "session": session,
                    "run": run,
                    "group": group,
                    "distances": distances,
                    "distance_type": distance_type,
                    "notes": notes if notes else "",
                }
            )

    distance_df = pd.DataFrame(rows)

    # Save DataFrame
    output_dir = "distance_analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{distance_type}_distances_{dataset_name}.pkl"
    filepath = os.path.join(output_dir, filename)

    print(f"Saving DataFrame to {filepath}")
    print(f"DataFrame shape: {distance_df.shape}")
    distance_df.to_pickle(filepath)

    return distance_df


def main():
    """
    Main processing function.
    """
    # Default parameters
    dataset_name = "OASIS"

    # Parse command line arguments
    if len(sys.argv) >= 2:
        dataset_name = sys.argv[1]

    print(f"Processing dataset: {dataset_name}")

    # List of distance types to compute
    distance_types = [
        "frobenius",
        "spectral",
        "nuclear",
        "manhattan",
        "chebyshev",
        "wasserstein_birth",
        "wasserstein_death",
    ]

    # Set data path (modify as needed)
    data_folder_path = f"connectivity_matrices/{dataset_name}"

    # Process each distance type
    for distance_type in distance_types:
        print(f"\n{'='*60}")
        print(f"Processing {distance_type} distance")
        print(f"{'='*60}")

        try:
            # Compute distances for all subjects
            distance_dict, subject_info_list = process_all_subjects(
                data_folder_path, distance_type
            )

            # Create DataFrame
            distance_df = create_distance_dataframe(
                dataset_name, subject_info_list, distance_dict, distance_type
            )

        except Exception as e:
            print(f"Error processing {distance_type} distance: {e}")
            import traceback

            traceback.print_exc()

    print("\nDistance analysis complete!")


if __name__ == "__main__":
    main()
