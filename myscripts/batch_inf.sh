#!/bin/bash

# Main variables
model_name="sub567_480"
CHECKPOINT_DIR="/workspace/public/users/lichang93/mydocker/cogvx/host_folder/Open-Sora-Plan/runs/$model_name"
declare -a specific_ckpts=()  # List of specific checkpoint numbers to evaluate

# Function to run evaluation using torchrun
run_evaluation() {
    local model_name=$1
    local ckpt_num=$2
    local model_path=$3
    local save_img_path="./eval/${model_name}_ckpt_${ckpt_num}"

    # Execute the torchrun command with provided arguments
    torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
             -m opensora.sample.sample_t2v_sp \
             --save_img_path "$save_img_path" \
             --model_path "$model_path" \
             --text_prompt 'myprompts/fridge_c6_5.txt' \
             --num_frames 93 \
             --height 480 \
             --width 640 \
             --cache_dir "./cache_dir" \
             --text_encoder_name "/workspace/host_folder/Open-Sora-Plan/google-mt5-xxl" \
             --ae CausalVAEModel_D4_4x8x8 \
             --ae_path "/workspace/public/models/Open-Sora-Plan-v1.2.0/vae" \
             --fps 24 \
             --guidance_scale 7.5 \
             --num_sampling_steps 100 \
             --enable_tiling \
             --max_sequence_length 512 \
             --sample_method DDPM \
             --model_type "dit"
}

# Function to check and run evaluations
function check_and_run {
    local ckpt_num=$1
    local ckpt_folder="$CHECKPOINT_DIR/checkpoint-$ckpt_num"
    if [ -d "$ckpt_folder" ]; then
        run_evaluation "$model_name" "$ckpt_num" "$ckpt_folder/model"
    else
        echo "Checkpoint directory $ckpt_folder does not exist."
    fi
}

# Check if specific checkpoints are provided
if [ ${#specific_ckpts[@]} -ne 0 ]; then
    # Evaluate specific checkpoints only
    for ckpt_num in "${specific_ckpts[@]}"; do
        check_and_run "$ckpt_num"
    done
else
    # Iterate over all checkpoint folders
    for ckpt_folder in $CHECKPOINT_DIR/checkpoint-*; do
        ckpt_num=$(basename $ckpt_folder | sed 's/checkpoint-//')
        check_and_run "$ckpt_num"
    done
fi
