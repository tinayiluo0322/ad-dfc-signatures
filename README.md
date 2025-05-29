# Topological and Geometric Signatures of Dynamic Functional Connectivity in Alzheimer’s Disease

This repository contains the code accompanying the paper:

**"Topological and Geometric Signatures of Dynamic Functional Connectivity in Alzheimer’s Disease: A Permutation-Based Framework"**

## Overview

This work proposes a novel permutation-based framework to identify topological and geometric biomarkers of Alzheimer’s disease (AD) using dynamic functional connectivity (dFC) derived from resting-state fMRI data.

The repository includes:

* ✅ **Data Preprocessing**
* ✅ **Dynamic Connectivity Matrix Construction**
* ✅ **Distance-Based Time Series Generation**
* ✅ **Permutation Testing Across Diagnostic Groups And Sexes**

## Folder Structure

```
├── Data-Preprocessing/
│   ├── fmriprep_simg.sh             # Runs fMRIPrep using Singularity
│   ├── generate_conn_matrices.py   # Generates connectivity matrices from preprocessed data
│   └── process_raw.sh              # Processes raw BIDS-formatted data
│
├── ConnectivityPipeline/
│   ├── extract_distance.py         # Computes distance metrics (e.g., Wasserstein, spectral)
│   └── Process_sliding_window.py   # Implements sliding-window dFC computation
│
├── PermutationTest/
│   ├── oasis_5peak_permutation.py  # Permutation test using peak-based features
│   └── oasis_mean_permutation.py   # Permutation test using mean-based features

```

## Requirements

This project relies on a combination of Python packages and neuroimaging tools for preprocessing, connectivity analysis, and statistical testing. It also includes SLURM job scripts for HPC environments.

### Python Packages

To install the core dependencies, run:

```bash
pip install -r requirements.txt
```

Or manually install the following:

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

> **Note:** Some functions also use `google.colab` (for Google Drive mounting) if running on Colab notebooks.

### Optional / Environment-Specific

* `fMRIPrep` (required for anatomical preprocessing)
* Install via Singularity or Docker depending on system
* SLURM workload manager (for HPC job scheduling)
* `bash` and job scripts using `SBATCH` directives for parallel processing

### System Requirements

* Python 3.8+
* At least 16–32 GB RAM recommended for full pipeline execution
* Access to BIDS-formatted fMRI data for preprocessing

## Citation

If you use this code, please cite:

> Yi, L., Tan, K., & Lutz, A. (2025). *Topological and Geometric Signatures of Dynamic Functional Connectivity in Alzheimer’s Disease: A Permutation-Based Framework*. \[Journal submission in progress]

## License

MIT License
