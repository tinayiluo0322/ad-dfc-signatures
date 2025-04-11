import os
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.interfaces.fmriprep import load_confounds
import numpy as np
import fnmatch
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt


def handle_nan_inf(data):
    """
    Handle NaN and Inf values in numpy arrays or pandas DataFrames
    """
    if isinstance(data, pd.DataFrame):
        return data.fillna(0).replace([np.inf, -np.inf], 0)
    elif isinstance(data, np.ndarray):
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return data
    return data


# Fetch atlas and setup masker
atlas = datasets.fetch_atlas_aal(version="SPM12")
masker = NiftiLabelsMasker(
    labels_img=atlas.maps,
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
    memory="nilearn_cache",
    verbose=5,
)

# Setup the connectivity measure
correlation_measure = ConnectivityMeasure(
    kind="correlation",
    standardize="zscore_sample",
)

# Directory containing all subjects
base_dir = "0_timeseries_confounds_new"

# Directory to save correlation matrices
corr_dir = "0_corr_matrices_new/correlation_matrices"
ts_dir = "0_corr_matrices_new/timeseries_npy"
plot_dir = "0_corr_matrices_new/correlation_matrix_plot"
os.makedirs(corr_dir, exist_ok=True)
os.makedirs(ts_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Function to get sorted list of subject folders
def get_sorted_subjects(base_dir):
    subjects = []
    for folder_name in os.listdir(base_dir):
        if fnmatch.fnmatch(folder_name, "sub-*"):
            subject_id = str(folder_name.split('-')[1])  # Extract the patient ID
            subjects.append(subject_id)
    return sorted(subjects)

# Function to find all corresponding FMRI files in ses-{i}/func folders
def find_fmri_files(subject_dir):
    fmri_files = []
    for root, dirs, files in os.walk(subject_dir):
        for file_name in files:
            if fnmatch.fnmatch(file_name, "*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"):
                fmri_files.append(os.path.join(root, file_name))
    return fmri_files if fmri_files else None

# Get sorted subjects list
sorted_subjects = get_sorted_subjects(base_dir)

# Loop through each sorted subject
for subject_id in sorted_subjects:
    subject_id_formatted = str(subject_id)
    subject_dir = os.path.join(base_dir, f"sub-{subject_id_formatted}")

    # Find all fmri files in ses-{i}/func
    fmri_files = find_fmri_files(subject_dir)
    
    if fmri_files:
        for fmri_path in fmri_files:
            # Extract session id (e.g., ses-{i}) from the file path
            ses_id = fmri_path.split('/')[-3]
            base_filename = os.path.basename(fmri_path).replace('.nii.gz', '').split('_space-MNI152NLin2009cAsym')[0]  # no extension

            # Load confounds
            confounds, sample_mask = load_confounds(
                fmri_path,
                strategy=["motion", "wm_csf"],
                motion="basic",
            )
            
            # Handle NaN and Inf in confounds
            if confounds is not None:
                confounds = handle_nan_inf(confounds)
            
            # Load NIfTI file to get the TR
            img = nib.load(fmri_path)
            tr = img.header.get_zooms()[3]  # Extracting the TR (Repetition Time)
            
            # Process the time series
            time_series = masker.fit_transform(
                fmri_path,
                confounds=confounds,
                sample_mask=sample_mask
            )
            time_series = handle_nan_inf(time_series)
        
            ######################
            # Save timeseries
            # Generate time stamps
            n_timepoints = time_series.shape[0]  # (n_timepoints, n_regions)
            time_stamps = np.array([tr * i for i in range(n_timepoints)])  # Time in seconds
            
            # Combine time stamps with each region's time series to create (n_timepoints, 2) arrays
            all_regions_time_series = np.stack([np.column_stack((time_stamps, time_series[:, region]))
                                                for region in range(time_series.shape[1])])
            # all_regions_time_series now has shape (num_regions, n_timepoints, 2)
            
            # Save the combined time series for all regions as .npy file
            npy_output_file = os.path.join(ts_dir, f"{base_filename}_time_series.npy")
            np.save(npy_output_file, all_regions_time_series)
        
        
            ######################
            # Save correlation matrix
            # Compute the correlation matrix
            correlation_matrix = correlation_measure.fit_transform([time_series])[0]
            correlation_matrix = handle_nan_inf(correlation_matrix)
            np.fill_diagonal(correlation_matrix, 0)
            
            # # Plot the correlation matrix
            # plotting.plot_matrix(
            #     correlation_matrix,
            #     figure=(10, 8),
            #     labels=atlas.labels,
            #     vmax=0.8,
            #     vmin=-0.8,
            #     title=f"Motion, WM-CSF - {os.path.basename(fmri_path)}",  # Use full .nii.gz file name in the title
            #     reorder=False,
            # )
        
            # # Save the plot to the plot_dir
            # plot_filename = f"sub-{subject_id_formatted}_{ses_id}_correlation_plot.png"
            # plot_path = os.path.join(plot_dir, plot_filename)
            # plt.savefig(plot_path, bbox_inches='tight')
            # plt.close()
            
            # print(f"Saved plot to {plot_path}")

            # Save the correlation matrix as a CSV file
            csv_filename = f"{base_filename}.csv"
            csv_path = os.path.join(corr_dir, csv_filename)
            np.savetxt(csv_path, correlation_matrix, delimiter=",")
            print(f"Saved correlation matrix to {csv_path}")

        else:
            print(f"FMRI file not found for subject {subject_id_formatted}")
