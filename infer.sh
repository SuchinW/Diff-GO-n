#!/bin/bash -l

#SBATCH -J sampling
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=72:00:00
#SBATCH --mem=32GB
#SBATCH --partition=partitions-gpu
#SBATCH --mail-user=user@email.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --nodelist=ruby-0
#SBATCH --gres=gpu:1  # Request 4 GPUs

# Load the CUDA module, if available
module load cuda

# Activate the Conda environment
source activate diff_go_n  # or conda activate diff if 'source' doesn't work

# Define log files for stdout and stderr
output_log="output_infer.log"
error_log="error_infer.log"

# Run your Python script and log stdout and stderr

python image_sample_noise_exp.py  --data_dir "path/to/dataset_folder"\
                                  --dataset_mode cityscapes \
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
                                  --num_classes 35 \
                                  --class_cond True \
                                  --no_instance False \
                                  --batch_size 1 \
                                  --num_samples 100 \
                                  --model_path ./checkpoints/ema_0.9999_100000.pt \
                                  --results_path ./results/results_100000 \
                                  --s 2 \
                                  --one_hot_label True \
                                  --snr 100 \
                                  --pool None \
                                  --unet_model unet
