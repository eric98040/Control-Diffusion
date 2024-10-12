# main.py

import os
import math
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torchvision import transforms
import json

# import h5py
import timeit

from models import Unet, ResNet34_embed, model_y2h
from diffusion import GaussianDiffusion
from trainer import Trainer
from opts import parse_opts
from utils import get_parameter_number, PowerWaveDataset
from train_net_for_label_embed import train_net_embed, train_net_y2h

args = parse_opts()

# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

if args.torch_model_path != "None":
    os.environ["TORCH_HOME"] = args.torch_model_path

# Output folders
path_to_output = os.path.join(args.root_path, "output")
os.makedirs(path_to_output, exist_ok=True)

path_to_embed_models = os.path.join(path_to_output, "embed_models")
os.makedirs(path_to_embed_models, exist_ok=True)

save_setting_folder = os.path.join(path_to_output, "{}".format(args.setting_name))
os.makedirs(save_setting_folder, exist_ok=True)

save_results_folder = os.path.join(save_setting_folder, "results")
os.makedirs(save_results_folder, exist_ok=True)


def main():
    # data loader
    dataset = PowerWaveDataset(
        design_folder=args.data_path + "/image",
        wave_path=args.data_path + "/0130_40_generated_wave_array2.csv",
        power_path=args.data_path + "/0130_40_generated_power_array2.csv",
    )

    print("\nRange of labels:")
    print(
        f"Power: ({np.min(dataset.labels[:, 0]):.4f}, {np.max(dataset.labels[:, 0]):.4f})"
    )
    print(
        f"Wave: ({np.min(dataset.labels[:, 1]):.4f}, {np.max(dataset.labels[:, 1]):.4f})"
    )

    # unique rows of labels: power and wavelength should both be unique
    unique_labels_norm = np.unique(
        dataset.labels, axis=0
    )  # axis=0: row scan ↓ (239972, 2)

    # vicinal parameters
    if args.kernel_sigma < 0:  # -1.0
        std_label = np.std(
            dataset.labels, axis=0
        )  # axis=0: row scan ↓, initial value: [0.24751157 0.28867984]
        args.kernel_sigma = (
            1.06 * np.mean(std_label) * (len(dataset.labels)) ** (-1 / 5)
        )
        print(f"\nKernel sigma: {args.kernel_sigma:.6f}")  # 0.023854

    if args.kappa < 0:  # -1.0
        n_unique = len(unique_labels_norm)  # 239972
        diff_list = [
            np.linalg.norm(unique_labels_norm[i] - unique_labels_norm[i - 1])
            for i in range(1, n_unique)  # i: 1 ~ 239971
        ]
        kappa_base = np.abs(args.kappa) * np.max(np.array(diff_list))

        args.kappa = kappa_base if args.threshold_type == "hard" else 1 / kappa_base**2

    # build training set for embedding network
    trainset_embedding = PowerWaveDataset(
        args.data_path + "/image",
        args.data_path + "/0130_40_generated_wave_array2.csv",
        args.data_path + "/0130_40_generated_power_array2.csv",
        normalize=False,
    )
    trainloader_embed_net = torch.utils.data.DataLoader(
        trainset_embedding,
        batch_size=args.batch_size_embed,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Embedding network
    net_embed_filename_ckpt = os.path.join(
        path_to_embed_models,
        "ckpt_{}_epoch_{}_seed_{}.pth".format(
            args.net_embed, args.epoch_cnn_embed, args.seed
        ),
    )
    net_y2h_filename_ckpt = os.path.join(
        path_to_embed_models,
        "ckpt_net_y2h_epoch_{}_seed_{}.pth".format(args.epoch_net_y2h, args.seed),
    )

    net_embed = ResNet34_embed(dim_embed=args.dim_embed)
    net_embed = net_embed.cuda()

    net_y2h = model_y2h(dim_embed=args.dim_embed)
    net_y2h = net_y2h.cuda()

    # Train net_embed
    if not os.path.isfile(net_embed_filename_ckpt):
        print("\n Start training CNN for label embedding >>>")
        net_embed = train_net_embed(
            net=net_embed,
            net_name=args.net_embed,
            trainloader=trainloader_embed_net,
            testloader=None,
            epochs=args.epoch_cnn_embed,
            resume_epoch=args.resumeepoch_cnn_embed,
            lr_base=0.01,
            lr_decay_factor=0.1,
            lr_decay_epochs=[80, 140],
            weight_decay=1e-4,
            path_to_ckpt=path_to_embed_models,
        )
        # save model
        torch.save(
            {
                "net_state_dict": net_embed.state_dict(),
            },
            net_embed_filename_ckpt,
        )
    else:
        print("\n net_embed ckpt already exists")
        print("\n Loading...")
        checkpoint = torch.load(net_embed_filename_ckpt)
        net_embed.load_state_dict(checkpoint["net_state_dict"])

    # Train y2h
    if not os.path.isfile(net_y2h_filename_ckpt):
        print("\n Start training net_y2h >>>")
        net_y2h = train_net_y2h(
            unique_labels_norm,
            net_y2h,
            net_embed,
            epochs=args.epoch_net_y2h,
            lr_base=0.01,
            lr_decay_factor=0.1,
            lr_decay_epochs=[150, 250, 350],
            weight_decay=1e-4,
            batch_size=128,
        )
        # save model
        torch.save(
            {
                "net_state_dict": net_y2h.state_dict(),
            },
            net_y2h_filename_ckpt,
        )
    else:
        print("\n net_y2h ckpt already exists")
        print("\n Loading...")
        checkpoint = torch.load(net_y2h_filename_ckpt)
        net_y2h.load_state_dict(checkpoint["net_state_dict"])

    # Build Unet
    attention_resolutions = [int(res) for res in args.attention_resolutions.split("_")]
    channel_mult = [int(c) for c in args.channel_mult.split("_")]

    model = Unet(
        embed_input_dim=args.dim_embed,
        cond_drop_prob=args.cond_drop_prob,
        in_channels=1,
        model_channels=args.model_channels,
        out_channels=None,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=0,
        channel_mult=channel_mult,
        conv_resample=True,
        num_heads=args.num_heads,
        use_scale_shift_norm=True,
        learned_variance=False,
        num_groups=args.num_groups,
    )
    model = nn.DataParallel(model)
    print("\r model size:", get_parameter_number(model))

    # Build diffusion process
    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.train_timesteps,
        sampling_timesteps=args.train_timesteps,
        objective=args.pred_objective,
        beta_schedule=args.beta_schedule,
        ddim_sampling_eta=1,
    ).cuda()

    # Prepare for training
    vicinal_params = {
        "kernel_sigma": args.kernel_sigma,
        "kappa": args.kappa,
        "threshold_type": args.threshold_type,
        "nonzero_soft_weight_threshold": args.nonzero_soft_weight_threshold,
    }

    num_samples = 16  # 시각화할 샘플 수
    y_visual = (
        torch.linspace(0, 1, num_samples).repeat(2, 1).t().cuda()
    )  # [num_samples, 2] 형태

    trainer = Trainer(
        diffusion_model=diffusion,
        train_dataset=dataset,
        vicinal_params=vicinal_params,
        train_batch_size=args.train_batch_size,
        gradient_accumulate_every=args.gradient_accumulate_every,
        train_lr=args.train_lr,
        train_num_steps=args.niters,
        ema_update_after_step=100,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        sample_every=args.sample_every,
        save_every=args.save_every,
        results_folder=save_results_folder,
        amp=args.train_amp,
        mixed_precision_type="fp16",
        split_batches=True,
        max_grad_norm=1.0,
        y_visual=y_visual,
        cond_scale_visual=6.0,
    )

    if args.resume_niter > 0:
        trainer.load(args.resume_niter)

    trainer.train(net_y2h=net_y2h)

    return trainer, net_y2h


if __name__ == "__main__":
    trainer, net_y2h = main()
    # 마지막 체크포인트 로드
    trainer.load(args.niters)
    # 모델을 평가 모드로 설정
    trainer.model.eval()
    # 샘플링 수행

    # 진행 상황을 저장할 파일
    progress_file = os.path.join(save_results_folder, "sampling_progress.json")

    # 이전 진행 상황 로드 또는 초기화
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)
        start_idx = progress["last_sampled_index"] + 1
        image_counter = progress["image_counter"]
    else:
        start_idx = 0
        image_counter = 1

    print(f"Generating samples using weights: model-{args.niters}.pt")

    # 모든 조건(레이블) 로드
    dataset = PowerWaveDataset(
        design_folder=args.data_path + "/image",
        wave_path=args.data_path + "/0130_40_generated_wave_array2.csv",
        power_path=args.data_path + "/0130_40_generated_power_array2.csv",
    )

    labels = [sample["labels"] for sample in dataset]
    labels = torch.stack(labels)  # Shape: [N, 2], N은 총 조건 수

    # generate (num_samples_per_condition) samples for each condition
    num_samples_per_condition = 4
    batch_size = args.samp_batch_size

    # 생성된 이미지를 저장할 디렉토리 생성
    output_dir = os.path.join(save_results_folder, "generated_samples")
    os.makedirs(output_dir, exist_ok=True)

    # 이미지 번호를 관리하기 위한 변수 추가
    image_counter = 1  # 시작 번호를 1로 설정

    for idx in tqdm(range(len(labels)), desc="Generating samples"):
        condition_label = labels[idx]
        condition_labels = condition_label.unsqueeze(0).repeat(
            num_samples_per_condition, 1
        )  # Shape: [4, 2]

        generated_images = trainer.sample_given_labels(
            given_labels=condition_labels,
            net_y2h=net_y2h,
            # batch_size=batch_size,
            batch_size=condition_labels.size(0),
            sampler=args.sampler,
            cond_scale=args.sample_cond_scale,
            sample_timesteps=args.sample_timesteps,
            ddim_eta=args.ddim_eta,
        )

        # 생성된 이미지를 저장
        for sample_idx, img in enumerate(generated_images):
            img = torch.clamp(img, 0, 1)  # 값 범위를 [0,1]로 제한
            save_path = os.path.join(
                output_dir, f"condition_{idx}_sample_{sample_idx}.tiff"
            )
            # 이미지 저장 (그레이스케일 모드)
            transforms.ToPILImage()(img.squeeze()).convert("L").save(save_path)
            image_counter += 1  # 이미지 번호 증가

    print(f"Generated samples saved to {output_dir}")
