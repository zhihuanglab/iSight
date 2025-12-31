#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

CONFIG_FILE="/XXXX/config/config.ini"
CHECKPOINT_PATH="/XXXX/XXX.pth"
INFERENCE_DATA="/XXXX/validation_data/validation_metadata.csv"
HDF5_BASE_DIR="/XXXX/validation_data/rle_masks"
RLE_MAP_PATH="/XXXX/validation_data/rle_masks/rle_mask_index.json"
OUTPUT_DIR="./results"


python inference.py \
  --config "$CONFIG_FILE" \
  --inference_data_path "$INFERENCE_DATA" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --hdf5_base_dir "$HDF5_BASE_DIR" \
  --rle_map_path "$RLE_MAP_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 4 \
  --num_workers 4 \
  --generate_visualizations \
  --save_name validation_results \
  --save_logits
