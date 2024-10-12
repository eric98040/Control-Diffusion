import numpy as np
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import os
import logging

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator

from ema_pytorch import EMA
from utils import (
    cycle,
    divisible_by,
    exists,
    hflip_images_tensor,
    normalize_images,
)
from moviepy.editor import ImageSequenceClip


# PIL을 사용하여 이미지 저장하는 함수
def save_image_grid_pil(
    gen_imgs, step, results_folder, normalize=True, nrow=4, mode="L"
):
    """
    배치 내 여러 이미지를 그리드 형태로 PIL을 사용하여 저장하는 함수
    gen_imgs: Tensor of shape [N, C, H, W]
    step: 현재 학습 스텝
    results_folder: 이미지가 저장될 폴더 경로
    normalize: True면 [-1,1]을 [0,1]로 변환, False면 [0,1]을 유지
    nrow: 한 줄에 배치할 이미지 수
    mode: 'L' for grayscale, 'RGB' for color
    """
    from math import ceil

    N, C, H, W = gen_imgs.shape
    grid_rows = ceil(N / nrow)
    grid_img = Image.new(mode, (W * nrow, H * grid_rows))

    for idx in range(N):
        # 텐서에서 배치 차원 제거
        img = gen_imgs[idx].detach().cpu()

        # Clamp to [0,1]
        img = torch.clamp(img, 0, 1)

        # 변환: [C, H, W] -> [H, W, C]
        img_np = img.permute(1, 2, 0).numpy()

        # 그레이스케일 이미지인 경우 채널 수 조정
        if mode == "L" and img_np.shape[2] == 1:
            img_np = img_np.squeeze(2)

        # [0,1]을 [0,255]로 변환
        img_np = (img_np * 255).astype(np.uint8)

        # PIL 이미지 생성
        pil_img = Image.fromarray(img_np, mode=mode)

        # 그리드에 이미지 붙이기
        row = idx // nrow
        col = idx % nrow
        grid_img.paste(pil_img, (col * W, row * H))

    # 저장 경로 설정
    save_path = Path(results_folder) / f"sample_{step}_grid.png"
    grid_img.save(save_path)

    tqdm.write(f"Saved image grid: {save_path}")


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        train_dataset,
        vicinal_params,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_after_step=1e30,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        sample_every=1000,
        save_every=1000,
        results_folder="./results",
        amp=False,
        mixed_precision_type="fp16",
        split_batches=True,
        max_grad_norm=1.0,
        y_visual=None,
        cond_scale_visual=6.0,
    ):
        super().__init__()

        # 데이터셋
        self.train_dataset = train_dataset
        self.train_labels = train_dataset.labels
        assert (
            self.train_labels.shape[1] == 2
        ), "Labels should have 2 dimensions (power and wave)"

        # vicinal params
        self.kernel_sigma = vicinal_params["kernel_sigma"]
        self.kappa = vicinal_params["kappa"]
        self.threshold_type = vicinal_params["threshold_type"]
        self.nonzero_soft_weight_threshold = vicinal_params[
            "nonzero_soft_weight_threshold"
        ]

        # 시각화
        self.y_visual = y_visual
        self.cond_scale_visual = cond_scale_visual

        # accelerator
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision_type if amp else "no"
        )

        # 모델
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # 샘플링 및 학습 하이퍼파라미터
        self.sample_every = sample_every
        self.save_every = save_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (
            train_batch_size * gradient_accumulate_every
        ) >= 16, f"your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above"

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # 옵티마이저
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # 결과를 주기적으로 저장하기 위한 EMA 설정
        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model,
                update_after_step=ema_update_after_step,
                beta=ema_decay,
                update_every=ema_update_every,
            )
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # 단계 카운터 상태
        self.step = 0

        # 모델, 옵티마이저를 accelerator로 준비
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": (
                self.accelerator.scaler.state_dict()
                if exists(self.accelerator.scaler)
                else None
            ),
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone, return_ema=False):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(
            str(self.results_folder / f"model-{milestone}.pt"), map_location=device
        )

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])
            if return_ema:
                return self.ema

        if "version" in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self, net_y2h):
        accelerator = self.accelerator
        device = accelerator.device

        ## label embedding network
        net_y2h = net_y2h.to(device)
        net_y2h.eval()

        log_filename = os.path.join(
            self.results_folder, "log_loss_niters{}.txt".format(self.train_num_steps)
        )
        if not os.path.isfile(log_filename):
            logging_file = open(log_filename, "w")
            logging_file.close()
        with open(log_filename, "a") as file:
            file.write(
                "\n==================================================================================================="
            )

        pbar = tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        )

        while self.step < self.train_num_steps:
            self.step += 1
            total_loss = 0.0

            for _ in range(self.gradient_accumulate_every):
                if self.threshold_type == "hard" and self.kappa == 0:
                    batch_real_indx = np.random.choice(
                        len(self.train_dataset), size=self.batch_size, replace=True
                    )
                    batch_images = torch.stack(
                        [self.train_dataset[i]["design"] for i in batch_real_indx]
                    )
                    batch_labels = torch.stack(
                        [self.train_dataset[i]["labels"] for i in batch_real_indx]
                    )

                    # 학습 루프에서 불필요한 정규화와 변환 제거
                    batch_images = batch_images.float().to(device)

                    # 수평 플립을 직접 텐서에서 수행
                    batch_images = hflip_images_tensor(batch_images)

                    with self.accelerator.autocast():
                        loss = self.model(
                            batch_images,
                            classes=net_y2h(batch_labels),
                            vicinal_weights=None,
                        )
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                ## vicinal loss 사용
                else:
                    ## 랜덤하게 batch_size개의 y를 train_labels에서 선택
                    batch_target_indices = np.random.choice(
                        len(self.train_labels), size=self.batch_size, replace=True
                    )
                    batch_target_labels = self.train_labels[batch_target_indices]

                    ## Gaussian 노이즈 추가
                    batch_epsilons = np.random.normal(
                        0, self.kernel_sigma, (self.batch_size, 2)
                    )
                    batch_target_labels = np.clip(
                        batch_target_labels + batch_epsilons, 0.0, 1.0
                    )

                    ## 실제 데이터에서 조건에 맞는 인덱스 찾기
                    batch_real_indx = np.zeros(
                        self.batch_size, dtype=int
                    )  # index of images in the data

                    for j in range(self.batch_size):
                        ## 조건에 맞는 인덱스 찾기
                        if self.threshold_type == "hard":
                            indx_real_in_vicinity = np.where(
                                np.all(
                                    np.abs(self.train_labels - batch_target_labels[j])
                                    <= self.kappa,
                                    axis=1,
                                )
                            )[0]
                        else:
                            indx_real_in_vicinity = np.where(
                                np.sum(
                                    (self.train_labels - batch_target_labels[j]) ** 2,
                                    axis=1,
                                )
                                <= -np.log(self.nonzero_soft_weight_threshold)
                                / self.kappa
                            )[0]

                        ## 조건에 맞는 데이터가 없을 경우 재샘플링
                        while len(indx_real_in_vicinity) < 1:
                            batch_epsilons_j = np.random.normal(0, self.kernel_sigma, 2)
                            batch_target_labels[j] = np.clip(
                                self.train_labels[
                                    np.random.choice(len(self.train_labels))
                                ]
                                + batch_epsilons_j,
                                0.0,
                                1.0,
                            )
                            ## 조건에 맞는 인덱스 재탐색
                            if self.threshold_type == "hard":
                                indx_real_in_vicinity = np.where(
                                    np.all(
                                        np.abs(
                                            self.train_labels - batch_target_labels[j]
                                        )
                                        <= self.kappa,
                                        axis=1,
                                    )
                                )[0]
                            else:
                                indx_real_in_vicinity = np.where(
                                    np.sum(
                                        (self.train_labels - batch_target_labels[j])
                                        ** 2,
                                        axis=1,
                                    )
                                    <= -np.log(self.nonzero_soft_weight_threshold)
                                    / self.kappa
                                )[0]

                        assert len(indx_real_in_vicinity) >= 1

                        batch_real_indx[j] = np.random.choice(
                            indx_real_in_vicinity, size=1
                        )[0]

                    ## 실제 이미지와 레이블 배치 생성
                    batch_target_labels = (
                        torch.from_numpy(batch_target_labels).float().to(device)
                    )
                    batch_images = torch.stack(
                        [self.train_dataset[i]["design"] for i in batch_real_indx]
                    )
                    batch_images = batch_images.float().to(device)
                    batch_images = hflip_images_tensor(batch_images)
                    batch_labels = self.train_labels[batch_real_indx]
                    batch_labels = torch.from_numpy(batch_labels).float().to(device)

                    ## weight vector 설정
                    if self.threshold_type == "soft":
                        vicinal_weights = torch.exp(
                            -self.kappa
                            * torch.sum(
                                (batch_labels - batch_target_labels) ** 2, dim=1
                            )
                        ).to(device)
                    else:
                        vicinal_weights = torch.ones(
                            self.batch_size, dtype=torch.float
                        ).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(
                            batch_images,
                            classes=net_y2h(batch_target_labels),
                            vicinal_weights=vicinal_weights,
                        )
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                self.accelerator.backward(loss)

            accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if self.step % 1000 == 0:
                with open(log_filename, "a") as file:
                    file.write(
                        "\r Step: {}, Loss: {:.4f}.".format(self.step, total_loss)
                    )

            accelerator.wait_for_everyone()

            self.opt.step()
            self.opt.zero_grad()

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                self.ema.update()

                if (
                    self.step != 0
                    and divisible_by(self.step, self.sample_every)
                    and self.y_visual is not None
                ):
                    self.ema.ema_model.eval()
                    with torch.inference_mode():
                        gen_imgs = self.ema.ema_model.ddim_sample(
                            classes=net_y2h(self.y_visual),
                            shape=(
                                self.y_visual.shape[0],
                                1,
                                self.image_size,
                                self.image_size,
                            ),
                            cond_scale=self.cond_scale_visual,
                            preset_sampling_timesteps=250,
                            preset_ddim_sampling_eta=0,
                        )

                        # 이미지 저장 함수 호출 (그리드 형태로 저장)
                        save_image_grid_pil(
                            gen_imgs,
                            self.step,
                            self.results_folder,
                            normalize=True,  # 정규화 상태에 따라 True 또는 False로 설정
                            nrow=4,  # 한 줄에 4개씩
                            mode="L",  # 그레이스케일 이미지
                        )

                if self.step != 0 and divisible_by(self.step, self.save_every):
                    milestone = self.step
                    self.ema.ema_model.eval()
                    self.save(milestone)

            pbar.set_description(f"loss: {total_loss:.4f}")
            pbar.update(1)

        pbar.close()
        accelerator.print("training complete")

    @torch.no_grad()
    def sample_given_labels(
        self,
        given_labels,
        net_y2h,
        batch_size=16,
        sampler="ddim",
        cond_scale=6.0,
        sample_timesteps=None,
        ddim_eta=0.0,
    ):
        self.model.eval()
        num_samples = given_labels.shape[0]
        generated_images = []
        for i in range(0, num_samples, batch_size):
            batch_labels = given_labels[i : i + batch_size]
            batch_classes = net_y2h(batch_labels.to(self.device))
            shape = (batch_classes.shape[0], 1, self.image_size, self.image_size)
            if sampler == "ddim":
                images = self.model.ddim_sample(
                    classes=batch_classes,
                    shape=shape,
                    cond_scale=cond_scale,
                    preset_sampling_timesteps=sample_timesteps,
                    preset_ddim_sampling_eta=ddim_eta,
                )
            else:
                images = self.model.p_sample_loop(
                    classes=batch_classes,
                    shape=shape,
                    cond_scale=cond_scale,
                    preset_sampling_timesteps=sample_timesteps,
                )
            generated_images.append(images.cpu())
        generated_images = torch.cat(generated_images, dim=0)
        return generated_images
