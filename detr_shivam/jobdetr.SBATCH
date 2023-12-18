#!/bin/bash
#SBATCH --job-name=detr1001gp
#SBATCH --open-mode=append
#SBATCH --output=./%x.out
#SBATCH --error=./%x.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --requeue

singularity exec --nv \
--overlay /scratch/sa7445/CV/cvenv.ext3:ro \
/scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
/bin/bash -c "source /ext3/env.sh; python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --resume /home/sa7445/sn-tracking-csgi2271/detr-r50_no-class-head.pth --coco_path /scratch/hk3820/ByteTrack/datasets/SoccerNet --batch_size 32  --num_classes 2 --epochs 50 --num_workers 16"
