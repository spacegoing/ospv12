# REAL_DATASET_DIR=/remote-home1/dataset/OpenMMLab___Kinetics-400/raw/Kinetics-400/videos_val/
REAL_DATASET_DIR=/workspace/host_folder/mycogvx/sat/pre/tmpbos/m5_6k/
EXP_NAME=decoder
SAMPLE_RATE=1
NUM_FRAMES=33
RESOLUTION=512
SUBSET_SIZE=6000
METRIC=ssim

python /workspace/Open-Sora-Plan/opensora/models/causalvideovae/eval/eval_common_metric.py \
    --batch_size 1 \
    --real_video_dir ${REAL_DATASET_DIR} \
    --generated_video_dir /workspace/Open-Sora-Plan/opensora/models/valid_gen/decoder_sr1_nf33_res512_subset6000 \
    --device cuda \
    --sample_fps 3 \
    --sample_rate ${SAMPLE_RATE} \
    --num_frames ${NUM_FRAMES} \
    --resolution ${RESOLUTION} \
    --subset_size ${SUBSET_SIZE} \
    --crop_size ${RESOLUTION} \
    --metric ${METRIC}
