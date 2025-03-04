#!/bin/bash -l

#SBATCH -J t_np_10k
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=72:00:00
#SBATCH --mem=64GB
#SBATCH --partition=partitions-gpu
#SBATCH --mail-user=user@email.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:1  # Request 4 GPUs

# Load the CUDA module, if available
module load cuda

# Activate the Conda environment
source activate diff_go_n  # or conda activate diff if 'source' doesn't work

# Define log files for stdout and stderr
output_log="output.log"
error_log="error.log"

# Run your Python script and log stdout and stderr
python image_train.py --data_dir "path/to/dataset_folder"\
                      --checkpoint_dir "./checkpoints"\
                      --dataset_mode cityscapes \
                      --lr 1e-4 \
                      --batch_size 1 \
                      --attention_resolutions 32,16,8 \
                      --diffusion_steps 1000 \
                      --image_size 256 \
                      --learn_sigma True \
                      --noise_schedule linear \
                      --num_channels 256 \
                      --num_head_channels 64 \
                      --num_res_blocks 2 \
                      --resblock_updown True \
                      --use_fp16 True \
                      --use_scale_shift_norm True \
                      --use_checkpoint True \
                      --save_interval 1000 \
                      --num_classes 35 \
                      --class_cond True \
                      --no_instance False \
                      --one_hot_label True
