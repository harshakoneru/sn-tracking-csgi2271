#!/bin/bash
#SBATCH --job-name=detrtrain
#SBATCH --open-mode=append
#SBATCH --output=./%x.out
#SBATCH --error=./%x.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --requeue

singularity exec --nv \
--overlay /scratch/sa7445/CV/cvenv.ext3:ro \
/scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
/bin/bash -c "source /ext3/env.sh; python3 main.py"
