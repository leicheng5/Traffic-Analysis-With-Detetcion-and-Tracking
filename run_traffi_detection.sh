#!/usr/bin/env bash
# ----------------------------------------------------
# set KMP_DUPLICATE_LIB_OK=TRUE
# chmod +x run_traffi_detection.sh
# ./run_traffi_detection.sh
# python main.py --source_video_path /path/to/your/video.mp4 --target_video_path /path/to/output.mp4 --source_weights_path /path/to/model/weights.pt --confidence_threshold 0.3 --iou_threshold 0.7 --display
# ----------------------------------------------------

# 1. (Optional) Activate your Conda environment
#    If not needed, comment out the next line
conda activate your_env_name

# 2. Define paths (update these to match your setup)
MODEL_PATH="/Users/username/projects/firedetection/best.pt"
VIDEO_INPUT="/Users/username/projects/firedetection/emniyett.mp4"
JSON_PATH="/Users/username/projects/firedetection/polygons.json"
VIDEO_OUTPUT="output_lane_detection.mp4"

# 3. Call the Python script
python main.py \
  --source_weights_path "${MODEL_PATH}" \
  --source_video_path "${VIDEO_INPUT}" \
  --target_video_path "${VIDEO_OUTPUT}" \
  --display \
  --confidence_threshold 0.1 \
  --iou_threshold 0.7

echo "=== Finished. Output saved to ${VIDEO_OUTPUT} ==="
