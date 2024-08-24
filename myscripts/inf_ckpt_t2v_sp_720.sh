torchrun --nnodes=1 --nproc_per_node 8  --master_port 29503 \
    -m opensora.sample.sample_t2v_sp \
    --model_path "/workspace/Open-Sora-Plan/runs/720_warmup1e5/checkpoint-1750/model" \
    --save_img_path "./eval/csp_720_warmup_1e5_1750" \
    --num_frames 93 \
    --height 720 \
    --width 1280 \
    --cache_dir "../cache_dir" \
    --text_encoder_name "/workspace/host_folder/Open-Sora-Plan/google-mt5-xxl" \
    --text_prompt examples/test_mt5.txt \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/workspace/public/models/Open-Sora-Plan-v1.2.0/vae" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit"
    # --sample_method DDPM \
    # --sample_method DPMSolverMultistep \
    # --tile_overlap_factor 0.125 \
