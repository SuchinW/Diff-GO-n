"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import torch as th
import torch.distributed as dist
import torchvision as tv

import guided_diffusion.gaussian_diffusion as gd
from guided_diffusion.image_datasets import load_data

# from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage, signal
from pooling import MedianPool2d
from PIL import Image, ImageFilter
import torchvision.utils as vutils
import torchvision.transforms as T

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SNR_DICT = {100: 0.0,
            30: 0.05,
            25: 0.08,
            20: 0.13,
            15: 0.22,
            10: 0.36,
            5: 0.6,
            1: 0.9}

noise_bag_size = 1000
th.manual_seed(150)
noise_bag = th.randn(noise_bag_size, 3, 256, 256)
th.manual_seed(np.random.randint(0, 1000000))

def main():
    args = create_argparser().parse_args()
    # dist_util.setup_dist()
    # logger.configure()
    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    device = "cuda:3"
    model.to(device)
    model.load_state_dict(th.load(args.model_path))
    model.to(device)

    print("creating data loader...")
    data = load_data(
        dataset_mode=args.dataset_mode,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        is_train=False
    )

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    image_path = os.path.join(args.results_path, 'images')
    os.makedirs(image_path, exist_ok=True)
    label_path = os.path.join(args.results_path, 'labels')
    os.makedirs(label_path, exist_ok=True)
    sample_path = os.path.join(args.results_path, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    noise_path = os.path.join(args.results_path, 'noises')
    os.makedirs(noise_path, exist_ok=True)

    print("sampling...")
    all_samples = []

    # device = "cuda:1"
    for i, (batch, cond) in enumerate(data):

        image = ((batch + 1.0) / 2.0).to(device)
        label = (cond['label_ori'].float() / 255.0).to(device)

        sample = image[0].cpu().numpy()
        sample = np.transpose(sample, (1, 2, 0))
        plot_label = cond['label'][0].cpu().numpy()
        plot_label = plot_label.squeeze(0)
        plot_label2 = cond['label_ori'][0].cpu().numpy()
        plot_label2 = plot_label2


        # model_kwargs = preprocess_input(args, cond, num_classes=args.num_classes, one_hot_label=args.one_hot_label, pool=None)
        model_kwargs = preprocess_input_FDS(args, cond, num_classes=args.num_classes, one_hot_label=args.one_hot_label,image=image)
        # model_kwargs, cond = preprocess_input(cond, one_hot_label=args.one_hot_label, add_noise=args.add_noise, noise_to=args.noise_to)

        # set hyperparameter
        model_kwargs['s'] = args.s

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )


        noise_schedule = "linear"

        step = 1000
        max_diffusion_steps = 1000
        batch_size = 1
        betas = gd.get_named_beta_schedule(noise_schedule, max_diffusion_steps)

        GD = gd.GaussianDiffusion(
            betas=betas,
            model_mean_type=gd.ModelMeanType.EPSILON,
            model_var_type=gd.ModelVarType.LEARNED_RANGE,
            loss_type=gd.LossType.MSE,
            rescale_timesteps=False
        )

        #Sampling starts from the forward diffusion process
        # t = th.tensor([step - 1] * batch_size, device=device)
        # noisy_input = GD.q_sample(batch.to(device), t)

        #Sampling starts from random noise from the noise bag
        random_index = th.randint(0, noise_bag_size, (1,)).item()
        noise = noise_bag[random_index].unsqueeze(0).to(device)

        sample = sample_fn(
            model,
            (args.batch_size, 3, image.shape[2], image.shape[3]),
            noise=noise, #noise=noisy_input,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True
        )

        sample = (sample + 1) / 2.0

        all_samples.extend([sample.cpu().numpy()])

        path = cond['path'][0].split('\\')[-1].split('.')[0] + '.png'

        s = path.split("/")[-1]
        tv.utils.save_image(image[0], os.path.join(image_path, 'rec_'+s))
        tv.utils.save_image(label[0], os.path.join(label_path, 'rec_'+s))
        for j in range(sample.shape[0]):
            tv.utils.save_image(sample[j],
                                os.path.join(sample_path, "n_step_" + str(step) + "_" + "r_step_1000_" + 'rec_'+s))
        print(f"created {len(all_samples) * args.batch_size} samples for the same input image")

    print("sampling complete")


def preprocess_input(args, data, num_classes, one_hot_label=True):
    # move to GPU and change data types
    data['label'] = data['label'].long()

    # create one-hot label map
    label_map = data['label']
    if one_hot_label:
        bs, _, h, w = label_map.size()
        input_label = th.FloatTensor(bs, num_classes, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        if 'instance' in data:
            inst_map = data['instance']
            instance_edge_map = get_edges(inst_map)
            input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)
    else:
        label_map = data['label']
        if 'instance' in data:
            # print("Instance in data")
            inst_map = data['instance']
            instance_edge_map = get_edges(inst_map)
            input_semantics = th.cat((label_map, instance_edge_map), dim=1)

    noise = th.randn(input_semantics.shape, device=input_semantics.device) * SNR_DICT[args.snr]
    input_semantics += noise
    print("Min, Mean, Max", th.min(input_semantics), th.mean(input_semantics), th.max(input_semantics))
    input_semantics = (input_semantics - th.min(input_semantics)) / (th.max(input_semantics) - th.min(input_semantics))
    print("Min, Mean, Max", th.min(input_semantics), th.mean(input_semantics), th.max(input_semantics))

    if args.pool == "med":
        print("Using Median filter")
        med_filter = MedianPool2d(padding=1, same=True)
        input_semantics_clean = med_filter(input_semantics)
    if args.pool == "mean":
        print("Using Average filter")
        avg_filter = th.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        input_semantics_clean = avg_filter(input_semantics)
    else:
        input_semantics_clean = input_semantics

    plt.figure(figsize=(30, 30))
    for idx, channel in enumerate(input_semantics_clean[0]):
        plt.subplot(6, 6, idx + 1)
        plt.imshow(channel.numpy(), cmap="gray")
        plt.axis("off")
    plt.savefig("./seg_map.png")

    return {'y': input_semantics_clean}


def preprocess_input_FDS(args, data, num_classes, one_hot_label=True, image=None):
    pool = None
    label_map = data['label'].long()
    if image is not (None):
        transform = T.ToPILImage()
        image = transform(image.squeeze(0))
        image = image.convert("L")
        image = image.filter(ImageFilter.FIND_EDGES)

    bs, _, h, w = label_map.size()
    input_label = th.FloatTensor(bs, num_classes, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    map_to_be_discarded = []
    map_to_be_preserved = []
    input_semantics = input_semantics.squeeze(0)
    
    for idx, segmap in enumerate(input_semantics.squeeze(0)):
        if 1 in segmap:
            map_to_be_preserved.append(idx)
        else:
            map_to_be_discarded.append(idx)

    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        transform = T.ToTensor()
        image = transform(image)  # image is the edge map
        instance_edge_map = instance_edge_map + image
        input_semantics = th.cat((input_semantics.unsqueeze(0), instance_edge_map), dim=1)

        map_to_be_preserved.append(num_classes)
        num_classes += 1

    print(input_semantics.shape, len(map_to_be_preserved))

    # input_semantics = input_semantics[map_to_be_preserved].unsqueeze(0)
    input_semantics = input_semantics[0][map_to_be_preserved]
    noise = th.randn(input_semantics.shape, device=input_semantics.device) * SNR_DICT[args.snr]

    input_semantics += noise

    if pool == "med":
        print("Using Median filter")
        med_filter = MedianPool2d(padding=1, same=True)
        input_semantics_clean = med_filter(input_semantics)
    elif pool == "mean":
        print("Using Average filter")
        avg_filter = th.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        # avg_filter2 = th.nn.AvgPool2d(kernel_size=5, stride=1, padding=1)
        input_semantics_clean = avg_filter(input_semantics)
    elif pool == "max":
        print("Using Max filter")
        avg_filter = th.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        max_filter = th.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        input_semantics_clean = max_filter(avg_filter(input_semantics))

    else:
        input_semantics_clean = input_semantics

    input_semantics_clean = input_semantics_clean.unsqueeze(0)

    # Insert non-classes maps
    input_semantics = th.empty(size=(input_semantics_clean.shape[0], \
                                     num_classes, input_semantics_clean.shape[2], \
                                     input_semantics_clean.shape[3]), device=input_semantics_clean.device)

    input_semantics[0][map_to_be_preserved] = input_semantics_clean[0]
    input_semantics[0][map_to_be_discarded] = th.zeros(
        (len(map_to_be_discarded), input_semantics_clean.shape[2], input_semantics_clean.shape[3]),
        device=input_semantics_clean.device)

    return {'y': input_semantics}


def calculate_activation_statistics(images, model, batch_size=128, dims=2048,
                                    cuda=False):
    model.eval()
    act = np.empty((len(images), dims))

    if cuda:
        batch = images.cuda()
    else:
        batch = images
    pred = model(batch)[0]
    print('pred shape 1: ', pred.shape)

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        print('pred shape 2: ', pred.shape)

    act = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="",
        results_path="",
        is_train=False,
        num_classes=35,
        s=1.0,
        snr=100,
        pool="med",
        add_noise=False,
        noise_to="semantics",
        unet_model="unet"  # "unet", "spadeboth", "enconly"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    print(parser)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
