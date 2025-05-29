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
├── preprocessing/            # fMRI data cleaning and formatting
├── connectivity_pipeline/    # Pearson correlation and sliding-window dFC
├── distance_metrics/         # Distance computation (e.g., Wasserstein, Spectral)
├── topology_analysis/        # Persistent graph homology and feature extraction
├── permutation_test/         # Group-level statistical testing pipeline
├── utils/                    # Helper functions and configurations
├── main.py                   # Entry script to run the pipeline
├── requirements.txt          # Dependencies
```

## Requirements

* Python 3.8+
* numpy, scipy, scikit-learn, gudhi, ripser, matplotlib, nibabel, nilearn

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the full pipeline:

```bash
python main.py --config config.yaml
```

Modify the config file to specify paths and analysis options.

## Citation

If you use this code, please cite:

> Yi, L., Tan, K., & Lutz, A. (2025). *Topological and Geometric Signatures of Dynamic Functional Connectivity in Alzheimer’s Disease: A Permutation-Based Framework*. \[Journal submission in progress]

## License

MIT License
