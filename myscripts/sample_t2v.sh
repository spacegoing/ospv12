CUDA_VISIBLE_DEVICES=0 python opensora/sample/sample_t2v.py \
    --save_img_path "./eval/single_720_euler" \
    --height 720 \
    --width 1280 \
    --model_path "/workspace/public/models/Open-Sora-Plan-v1.2.0/93x720p/" \
    # --save_img_path "./eval/single_480_ddpm" \
    # --height 480 \
    # --width 640 \
    # --model_path "/workspace/public/models/Open-Sora-Plan-v1.2.0/93x480p/checkpoint-12000/model/" \
    --text_prompt examples/test_mt5.txt \
    --num_frames 93 \
    --cache_dir "../cache_dir" \
    --text_encoder_name "/workspace/host_folder/Open-Sora-Plan/google-mt5-xxl" \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/workspace/public/models/Open-Sora-Plan-v1.2.0/vae" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --max_sequence_length 512 \
    # --sample_method DPMSolverMultistep \
    # --sample_method DDPM \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit"
