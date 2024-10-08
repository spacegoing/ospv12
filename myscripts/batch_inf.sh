#!/bin/bash

# Main variables
model_name="aes_45"
FOLDER="/workspace/public/users/lichang93/mydocker/cogvx/host_folder/Open-Sora-Plan"
CHECKPOINT_DIR="$FOLDER/runs/$model_name"
declare -a specific_ckpts=()  # List of specific checkpoint numbers to evaluate
declare -a processed_ckpts=() # List to keep track of processed checkpoints
start_idx=98500

# Function to run evaluation using torchrun
run_evaluation() {
    local model_name=$1
    local ckpt_num=$2
    local model_path=$3
    local save_img_path="${FOLDER}/eval/${model_name}/${model_name}_ckpt_${ckpt_num}"

    # Execute the torchrun command with provided arguments
    torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
             -m opensora.sample.sample_t2v_sp \
             --save_img_path "$save_img_path" \
             --model_path "$model_path" \
             --text_prompt 'myprompts/midhard.txt' \
             --num_frames 93 \
             --height 480 \
             --width 640 \
             --cache_dir "./cache_dir" \
             --text_encoder_name "${FOLDER}/google-mt5-xxl" \
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
    if [[ $ckpt_num -ge $start_idx ]] && [ -d "$ckpt_folder" ] && [[ ! " ${processed_ckpts[@]} " =~ " $ckpt_num " ]]; then
        run_evaluation "$model_name" "$ckpt_num" "$ckpt_folder/model"
        processed_ckpts+=("$ckpt_num")  # Mark this checkpoint as processed
    else
        echo "Checkpoint directory $ckpt_folder does not exist or already processed."
    fi
}

# Main loop to continuously monitor for new checkpoints
while true; do
    # for ckpt_folder in $CHECKPOINT_DIR/checkpoint-*; do
    for ckpt_folder in $(ls -d $CHECKPOINT_DIR/checkpoint-* 2>/dev/null | sort -V); do
        ckpt_num=$(basename $ckpt_folder | sed 's/checkpoint-//')
        check_and_run "$ckpt_num"
    done
    sleep 600  # Check every 10 minutes
done
