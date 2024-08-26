# export WANDB_MODE="offline"
export ENTITY="linbin"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PDSH_RCMD_TYPE=ssh
export XDG_CACHE_HOME="/workspace/public/users/lichang93/.cache"
export HF_HOME="/workspace/public/users/lichang93/.cache/huggingface"
# NCCL setting
# export GLOO_SOCKET_IFNAME=bond0
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TC=162
# export NCCL_IB_TIMEOUT=22
# export NCCL_PXN_DISABLE=0
# export NCCL_IB_QPS_PER_CONNECTION=4
# export NCCL_ALGO=Ring
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export NCCL_ALGO=Tree

accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    opensora/train/train_t2v_diffusers.py \
    --sp_size 8 \
    --train_sp_batch_size 2 \
    --output_dir="/workspace/Open-Sora-Plan/runs/480p_warm100_1e4/" \
    --data "m5.txt" \
    --dataset t2v \
    --cache_dir "./cache_dir" \
    --pretrained "/workspace/public/models/Open-Sora-Plan-v1.2.0/93x480p/diffusion_pytorch_model.safetensors" \
    --model OpenSoraT2V-ROPE-L/122 \
    --sample_rate 1 \
    --num_frames 93 \
    --max_height 480 \
    --max_width 640 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=100 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=250 \
    --allow_tf32 \
    --model_max_length 512 \
    --use_image_num 0 \
    --tile_overlap_factor 0.125 \
    --snr_gamma 5.0 \
    --use_ema \
    --ema_start_step 0 \
    --cfg 0.1 \
    --noise_offset 0.02 \
    --use_rope \
    --ema_decay 0.999 \
    --enable_tiling \
    --speed_factor 1.0 \
    --group_frame \
    --text_encoder_name "/workspace/public/users/lichang93/mydocker/cogvx/host_folder/Open-Sora-Plan/google-mt5-xxl" \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/workspace/public/models/Open-Sora-Plan-v1.2.0/vae"
    # --resume_from_checkpoint="latest" \
