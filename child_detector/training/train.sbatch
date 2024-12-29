#!/bin/bash
#SBATCH --partition rtx4090
#SBATCH --qos=azencot
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=6
#SBATCH --time 10-15:00:00
#SBATCH --job-name asdhub_detection
#SBATCH --output ./logs/job-%A_%a-%J.out
#SBATCH --error ./logs/job-%A_%a-%J.err
##SBATCH --mail-user=baramit@post.bgu.ac.il
##SBATCH --mail-type=NONE

mount /home/baramit/win-mount/Users
mount /home/baramit/win-mount/recordings

echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

source /home/baramit/miniconda3/bin/activate asdhub
cd /home/baramit/research/Child-Detector

python child_detector/training/train.py