# Topological and geometric signatures of brain network dynamics in Alzheimer’s Disease

This repository contains the code accompanying the paper:

***Topological and geometric signatures of brain network dynamics in Alzheimer’s Disease***

## Overview

This work presents a novel permutation-based framework to identify topological and geometric biomarkers of Alzheimer’s disease (AD) using dynamic functional connectivity (dFC) derived from resting-state fMRI data.

The repository includes:

* ✅ **Data Preprocessing**
* ✅ **Dynamic Connectivity Matrix Construction**
* ✅ **Distance-Based Time Series Generation**
* ✅ **Permutation Testing Across Diagnostic Groups and Sexes**

## Folder Structure

```
├── Data-Preprocessing/
│   ├── fmriprep_simg.sh             # Runs fMRIPrep using Singularity
│   ├── generate_conn_matrices.py    # Generates connectivity matrices from preprocessed data
│   └── process_raw.sh               # Processes raw BIDS-formatted data
│
├── ConnectivityPipeline/
│   ├── extract_distance.py          # Computes distance metrics (e.g., Wasserstein, spectral)
│   └── Process_sliding_window.py    # Implements sliding-window dFC computation
│
├── PermutationTest/
│   ├── oasis_5peak_permutation.py   # Permutation test using 5-peak-based features
│   └── oasis_mean_permutation.py    # Permutation test using mean-based features
│
├── requirements.txt                 # Python dependencies
```

## Requirements

This project relies on a combination of Python packages and neuroimaging tools for preprocessing, connectivity analysis, and statistical testing. SLURM job scripts are included for HPC environments.

### Python Packages

To install all core dependencies:

```bash
pip install -r requirements.txt
```

Or install them individually:

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scipy`
* `scikit-learn`
* `tqdm`
* `nibabel`
* `nilearn`
* `networkx`

> **Note:** Some scripts also use `google.colab` for mounting Google Drive.

### Optional / Environment-Specific Tools

* `fMRIPrep` (required for anatomical preprocessing)
* Install via **Singularity** or **Docker**
* `SLURM` workload manager (for HPC job scheduling)
* Shell (`bash`) and `SBATCH` scripts for parallel job submission

### System Requirements

* Python 3.8+
* 16–32 GB RAM recommended for full pipeline execution
* Access to BIDS-formatted resting-state fMRI data

## Citation

If you use this code, please cite:

> **Luopeiwen Yi**, **Michael William Lutz**\*, **Yutong Wu**, **Yang Li**, and **Tananun Songdechakraiwut**\* (2025).
> *Topological and geometric signatures of brain network dynamics in Alzheimer’s Disease*.
> \[Journal submission in progress]

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
