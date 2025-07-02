#!/usr/bin/bash

#SBATCH --qos=ng
#SBATCH --gpus=1
#SBATCH --time=10:00

# Set the environment
module load apptainer
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Get image paths as an array
image_paths=($(find UFDD_val/images -type f -and -not -wholename '*labelled*'))

# Run the detection algorithm on each of them
for image_path in ${image_paths[@]};
do
    apptainer exec --nv face-gpu.sif python3 face-gpu.py ${image_path}
done
