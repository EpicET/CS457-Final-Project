#!/usr/bin/env bash
# slurm template for serial jobs
#SBATCH --job-name=final_base # Job name
#SBATCH --output=final_base-%j.out
#SBATCH --mail-user=etowner@middlebury.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=4g
#SBATCH --partition=gpu-standard
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# print SLURM envirionment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" echo "Starting: "`date +"%D %T"` 
# Your calculations here 
python BERTtrain.py
# End of job info 
echo "Ending: "`date +"%D %T"`