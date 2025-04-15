#!/bin/bash
#SBATCH --job-name=massspecformer
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=massspecformer_%j.out

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

echo "Starting job on $(hostname) at $(date)"
python /gpfs/home2/mlotfiandeloue/datasets/mass\ spectra/1.Step1DRIAMSD_Snellius.py
echo "Job completed at $(date)"
