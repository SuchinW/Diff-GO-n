"""
Train a diffusion model on images.
"""

import argparse

# from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

# import wandb

device = "cuda"
print("Using device:", device)


def main():
    args = create_argparser().parse_args()

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(device)

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    print("creating data loader...")
    data = load_data(
        dataset_mode=args.dataset_mode,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        is_train=args.is_train
    )

    print("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        drop_rate=args.drop_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_dir=args.checkpoint_dir,
        one_hot_label=args.one_hot_label,
        snr=args.snr,
        add_noise=args.add_noise,
        noise_to=args.noise_to,
        unet_model=args.unet_model
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        drop_rate=0.0,
        log_interval=10,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        is_train=True,
        checkpoint_dir="./checkpoints",
        one_hot_label=True,
        snr=100,
        add_noise=False,
        noise_to="semantics",
        unet_model="unet"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
