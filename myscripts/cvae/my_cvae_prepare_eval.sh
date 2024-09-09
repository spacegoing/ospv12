export CUDA_VISIBLE_DEVICES=0
REAL_DATASET_DIR=/workspace/host_folder/mycogvx/sat/pre/tmpbos/m5_6k/
EXP_NAME=decoder
SAMPLE_RATE=1
NUM_FRAMES=33
RESOLUTION=512
SUBSET_SIZE=6000
CKPT=/workspace/public/models/Open-Sora-Plan-v1.2.0/vae

python causalvideovae/sample/rec_video_vae.py \
    --batch_size 1 \
    --real_video_dir ${REAL_DATASET_DIR} \
    --generated_video_dir valid_gen/${EXP_NAME}_sr${SAMPLE_RATE}_nf${NUM_FRAMES}_res${RESOLUTION}_subset${SUBSET_SIZE} \
    --device cuda \
    --sample_fps 24 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --subset_size ${SUBSET_SIZE} \
    --crop_size ${RESOLUTION} \
    --num_workers 8 \
    --ckpt ${CKPT} \
    --output_origin \
    --enable_tiling
