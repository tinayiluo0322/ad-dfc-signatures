#!/bin/bash
#SBATCH --job-name=process_row  # Job name
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --cpus-per-task=4  # Number of CPU cores per task
#SBATCH --time=120:00:00  # Maximum runtime
#SBATCH --mem=16G  # Memory allocation
#SBATCH --array=1-1400%400

BIDSDIR=OASIS/raw_data  # Input BIDS directory
DERIVSDIR=OASIS/preprocessed_data  # Output directory for derivatives
FSlicense=OASIS/license.txt  # Path to FreeSurfer license
RANDOMSEED=15  # Set a fixed random seed for reproducibility

# Get the list of all 'sub-' directories in the BIDS folder
subdirs=$(ls $BIDSDIR | grep 'sub-')

# Select the subdirectories for the current array task
selected_subdirs=$(echo "$subdirs" | sed -n "${SLURM_ARRAY_TASK_ID}p")

# Loop through each selected subject and process them
for subdir in $selected_subdirs; do
    subid=${subdir/sub-/}  # Extract the subject ID by removing 'sub-' prefix
    echo "Processing $subid"

    # Run fMRIPrep with Singularity for the selected subject
    singularity run --cleanenv --bind OASIS/:OASIS \
        OASIS/fmriprep.simg \
        $BIDSDIR $DERIVSDIR participant --participant-label $subid \
        --skip_bids_validation \
        --fs-no-reconall \
        --notrack \
        --output-spaces {T1w,MNI152NLin2009cAsym,func} \
        --fs-license-file $FSlicense \
        --skull-strip-fixed-seed \
        --random-seed $RANDOMSEED \
        --omp-nthreads 1 \
        --stop-on-first-crash

    echo "Finished processing $subid"
done
