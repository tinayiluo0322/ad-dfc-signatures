#!/bin/bash
#SBATCH --job-name=fmriprep_build
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00    
#SBATCH --mem=32G

singularity build fmriprep.simg docker://nipreps/fmriprep:latest