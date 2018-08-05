#!/bin/bash

#SBATCH --job-name label
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --time 00:15:00
#SBATCH --partition gpu
#SBATCH --output label.out
#SBATCH --memory 64G

module load CUDA
#module load /mnt/storage/home/xh17500/anaconda3

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOB_ID
echo This job runs on the following machines:
echo `echo $SLURM_JOB_NODELIST | uniq`

#! Run the executable
python ./assign_label.py
