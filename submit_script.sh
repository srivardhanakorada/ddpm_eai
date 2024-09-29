#!/bin/bash
#SBATCH --job-name=ddpm_mnist_job          # Job name
#SBATCH --output=ddpm_mnist_output.txt         # Standard output and error log
#SBATCH --ntasks=4                         # Number of tasks (processes)
#SBATCH --time=2:00:00                    # Time limit hrs:min:sec
#SBATCH --mem=10G                           # Total memory limit

source /raid/home/rizwank/miniconda3/etc/profile.d/conda.sh
conda activate diff_env             
python -m tools.train_ddpm                     
conda deactivate