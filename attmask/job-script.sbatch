#!/bin/bash -l
#SBATCH --nodes=1                  # Resource requirements, job runtime, other options 
#SBATCH --ntasks=1                 #All #SBATCH lines have to follow uninterrupted
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00            
#SBATCH --job-name=attmask-job 
#SBATCH --export=NONE
#SBATCH --mail-user=bastian.schaffer@fau.de
#SBATCH --mail-type=ALL

unset SLURM_EXPORT_ENV

module add python
export WORK=/home/woody/iwi5/iwi5266h
export PYTHONUSERBASE=$WORK/software/private
export OMP_NUM_THREADS=1
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export WANDB_DIR=$WORK/wandb
export PATH=$WORK/software/private/bin:$PATH

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900
export NCCL_DEBUG=INFO

srun python3 -m torch.distributed.run --nproc_per_node=4 main_attmask.py --batch_size_per_gpu 30 --num_workers 10 \
--arch vit_small --warmup_epochs 2 --warmup_teacher_temp_epochs 2 --epochs 20 --lr 0.005 \
--local_crops_number 8 --global_crops_scale 0.4 1 \
--local_crops_scale 0.05 0.4 --pred_shape attmask_high --freeze_last_layer 1 --num_channels 1 \
--data_path /home/woody/iwi5/iwi5266h/datasets/icdar2017-training-binary --output_dir /home/woody/iwi5/iwi5266h/training-output #\
#--load_from /home/woody/iwi5/iwi5266h/training-output/checkpoint.pth
