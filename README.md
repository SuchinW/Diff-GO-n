# Diff-GO<sup>n</sup>: Enhancing Diffusion Models for Goal-Oriented Communications

### Suchinthaka Wanninayaka, Achintha Wijesinghe, Weiwei Wang, Yu-Chieh Chao, Songyang Zhang, and Zhi Ding
[[ArXiv Preprint](https://arxiv.org/abs/2412.06980)]

📢 **Update**: Stay tuned for the release of pretrained models and implementation details!  

---

## 📄 Abstract  
The rapid expansion of edge devices and Internet-of-Things (IoT) continues to heighten the demand for data transport under limited spectrum resources. The goal-oriented communications (GO-COM), unlike traditional communication systems designed for bit-level accuracy, prioritizes more critical information for specific application goals at the receiver. To improve the efficiency of generative learning models for GO-COM, this work introduces a novel noise-restricted diffusion-based GO-COM (Diff-GO<sup>n</sup>) framework for reducing bandwidth overhead while preserving the media quality at the receiver. Specifically, we propose an innovative Noise-Restricted Forward Diffusion (NR-FD) framework to accelerate model training and reduce the computation burden for diffusion-based GO-COMs by leveraging a pre-sampled pseudo-random noise bank (NB). Moreover, we design an early stopping criterion for improving computational efficiency and convergence speed, allowing high-quality generation in fewer training steps. Our experimental results demonstrate superior perceptual quality of data transmission at a reduced bandwidth usage and lower computation, making Diff-GO<sup>n</sup> well-suited for real-time communications and downstream applications.

---

## 🛠️ The Diff-GO<sup>n</sup> Framework 
![Architecture](architechture.png)  

---

## 🚀 How to Install

### Step 1: Create a Conda Environment
To ensure a clean and isolated environment, create a new Conda environment named **`diff_go_n`**:

```bash
conda create --name diff_go_n python=3.8 -y
conda activate diff_go_n
```



### Step 2: Install Additional Dependencies
After installing PyTorch, install the remaining dependencies:

```bash
pip install -r requirements.txt
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

---
## 🖥️ Running on SLURM

Before submitting jobs, ensure you update the following SLURM parameters in the scripts:

- `#SBATCH --partition=partitions-gpu` → Set this to the appropriate GPU partition available on your cluster.
- `#SBATCH --mail-user=user@email.com` → Replace with your email to receive job status notifications.
- `--data_dir "path/to/dataset_folder"` → Specify the correct path to your dataset directory.

Refer to your cluster's documentation for specific configurations.

The training and inference processes in **Diff-GO<sup>n</sup>** are designed to be executed on SLURM-managed clusters. We provide SLURM scripts to facilitate job scheduling and execution.

### 🔥 Training on SLURM

#### **1️⃣ Submitting a Training Job**
To train the model on a SLURM cluster, use the `train.sh` script:

```bash
sbatch train.sh
```

Alternatively, you can run the training script directly without SLURM using the following command:

```bash
python image_train.py --data_dir "path/to/dataset_folder" \
                      --checkpoint_dir "./checkpoints" \
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
```

### 🎨 Sampling on SLURM

#### **2️⃣ Submitting a Sampling Job**
Once the model is trained, you can generate samples using the `sample.sh` script:

```bash
sbatch sample.sh
```

This script will execute the sampling process on the SLURM cluster, generating output based on the trained model.

Alternatively, you can run the sampling script directly without SLURM using the following command:

```bash
python image_sample.py --data_dir "path/to/dataset_folder" \
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
```

### 📥 Downloading Pretrained Model
If you want to use a pretrained model, you can download it from the following link:

[Pretrained Model](https://drive.google.com/file/d/1eGBopLWfVAO__7fZ7cRJVwl-iEl3Vtoy/view?usp=drive_link)

After downloading, create a folder named `checkpoints` and save the model inside it:

```bash
mkdir -p checkpoints
mv path/to/downloaded_model.pt checkpoints/
```


---
This repository is under construction! 🛠️ Stay tuned for more updates!
## 🎓 Acknowledgements
This project is based on [GESCO](https://github.com/ispamm/GESCO/), which was originally developed by ISPAMM Research Lab.

## Cite
If you find this work useful, please consider citing our paper:

```bibtex
@article{wanninayaka2024diff,
  title={Diff-GO $\^{}$\backslash$text $\{$n$\}$ $: Enhancing Diffusion Models for Goal-Oriented Communications},
  author={Wanninayaka, Suchinthaka and Wijesinghe, Achintha and Wang, Weiwei and Chao, Yu-Chieh and Zhang, Songyang and Ding, Zhi},
  journal={arXiv preprint arXiv:2412.06980},
  year={2024}
}
```
```