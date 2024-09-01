# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""

import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Optional
import gc
import numpy as np
from einops import rearrange
from tqdm import tqdm

from opensora.adaptor.modules import replace_with_fp32_forwards

try:
    import torch_npu
    from opensora.npu_config import npu_config
    from opensora.acceleration.parallel_states import (
        initialize_sequence_parallel_state,
        destroy_sequence_parallel_group,
        get_sequence_parallel_state,
        set_sequence_parallel_state,
    )
    from opensora.acceleration.communications import (
        prepare_parallel_data,
        broadcast,
    )
except:
    torch_npu = None
    npu_config = None
    from opensora.utils.parallel_states import (
        initialize_sequence_parallel_state,
        destroy_sequence_parallel_group,
        get_sequence_parallel_state,
        set_sequence_parallel_state,
    )
    from opensora.utils.communications import (
        prepare_parallel_data,
        broadcast,
    )

    pass
import time
from dataclasses import field, dataclass
from torch.utils.data import DataLoader
from copy import deepcopy
import accelerate
import torch
from torch.nn import functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedType,
    ProjectConfiguration,
    set_seed,
)
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    DDPMScheduler,
    PNDMScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available

from opensora.utils.ema import EMAModel
from opensora.dataset import getdataset
from opensora.models import CausalVAEModelWrapper
from opensora.models.text_encoder import (
    get_text_enc,
    get_text_warpper,
)
from opensora.models.causalvideovae import (
    ae_stride_config,
    ae_channel_config,
)
from opensora.models.causalvideovae import ae_norm, ae_denorm
from opensora.models.diffusion import (
    Diffusion_models,
    Diffusion_models_class,
)
from opensora.utils.dataset_utils import (
    Collate,
    LengthGroupedSampler,
)
from opensora.sample.pipeline_opensora import OpenSoraPipeline

import torch.distributed as dist


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")
logger = get_logger(__name__)


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    # use LayerNorm, GeLu, SiLu always as fp32 mode
    if args.enable_stable_fp32:
        replace_with_fp32_forwards()
    if torch_npu is not None and npu_config is not None:
        npu_config.print_msg(args)
        npu_config.seed_everything(args.seed)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.num_frames != 1 and args.use_image_num == 0:
        initialize_sequence_parallel_state(args.sp_size)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now. On GPU...
    if torch_npu is None and npu_config is None and args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Setup data:
    import json
    import argparse

    # Read the JSON file back into a dictionary
    with open('mytest/dataset_init_sp8_dp2_args.json', 'r') as f:
        args_dict = json.load(f)

    # Convert the dictionary back to an argparse.Namespace object
    args = argparse.Namespace(**args_dict)

    train_dataset = getdataset(args)
    sampler = (
        LengthGroupedSampler(
            args.train_batch_size,
            world_size=8,
            lengths=train_dataset.lengths,
            group_frame=args.group_frame,
            group_resolution=args.group_resolution,
        )
        if (args.group_frame or args.group_resolution)
        else None
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=sampler is None,
        # pin_memory=True,
        collate_fn=Collate(args),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        sampler=sampler if (args.group_frame or args.group_resolution) else None,
        drop_last=True,
        # prefetch_factor=4
    )

    train_dataloader = accelerator.prepare(train_dataloader)
    for step, data_item in enumerate(train_dataloader):
        # if train_one_step(step, data_item, prof_):
        #     break
        dist.breakpoint(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset & dataloader
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data", type=str, required="")
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--train_fps", type=int, default=24)
    parser.add_argument("--drop_short_ratio", type=float, default=1.0)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--max_height", type=int, default=320)
    parser.add_argument("--max_width", type=int, default=240)
    parser.add_argument("--use_img_from_vid", action="store_true")
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--cfg", type=float, default=0.1)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--group_frame", action="store_true")
    parser.add_argument("--group_resolution", action="store_true")

    # text encoder & vae & diffusion model
    parser.add_argument(
        "--model",
        type=str,
        choices=list(Diffusion_models.keys()),
        default="Latte-XL/122",
    )
    parser.add_argument("--enable_8bit_t5", action="store_true")
    parser.add_argument("--tile_overlap_factor", type=float, default=0.125)
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument("--compress_kv", action="store_true")
    parser.add_argument(
        "--attention_mode",
        type=str,
        choices=["xformers", "math", "flash"],
        default="xformers",
    )
    parser.add_argument("--use_rope", action="store_true")
    parser.add_argument("--compress_kv_factor", type=int, default=1)
    parser.add_argument("--interpolation_scale_h", type=float, default=1.0)
    parser.add_argument("--interpolation_scale_w", type=float, default=1.0)
    parser.add_argument("--interpolation_scale_t", type=float, default=1.0)
    parser.add_argument("--downsampler", type=str, default=None)
    parser.add_argument("--ae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--ae_path", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument(
        "--text_encoder_name",
        type=str,
        default="DeepFloyd/t5-v1_1-xxl",
    )
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--enable_stable_fp32", action="store_true")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    # diffusion setting
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use EMA model.",
    )
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0.02,
        help="The scale of noise offset.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )

    # validation & logs
    parser.add_argument("--num_sampling_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--enable_tracker", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # optimizer & scheduler
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamW",
        help='The optimizer type to use. Choose between ["AdamW", "prodigy"]',
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
        help="Use AdamW style decoupled weight decay",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-02,
        help="Weight decay to use for unet params",
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder",
        type=float,
        default=None,
        help="Weight decay to use for text_encoder",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--sp_size",
        type=int,
        default=1,
        help="For sequence parallel",
    )
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    args = parser.parse_args()
    main(args)
