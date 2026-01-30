# Run summary

- run_id: `convnext_tiny_sr_v1_260130_104646`
- started: `2026-01-30T10:46:46+07:00`
- ended: `2026-01-30T13:19:39+07:00`
- duration_sec: `9173.55`
- command: `train.py --model restran --backbone convnext --sr-aux --batch-size 8 --epochs 30 --experiment-name convnext_tiny_sr_v1`

## Config
- EXPERIMENT_NAME: `convnext_tiny_sr_v1`
- MODEL_TYPE: `restran`
- USE_STN: `True`
- DATA_ROOT: `data/train`
- EPOCHS: `30`
- BATCH_SIZE: `8`
- LEARNING_RATE: `0.0005`
- IMG_HEIGHT: `32`
- IMG_WIDTH: `128`
- AUGMENTATION_LEVEL: `full`
- VAL_SPLIT_FILE: `data/val_tracks.json`
- DEVICE: `cuda`

## Dataset
- train_size: `38002`
- val_size: `999`
- submission_mode: `False`
- data_root: `data/train`
- test_data_root: `data/public_test`

## Model
- type: `restran`
- use_stn: `True`
- total_params: `39521231`
- trainable_params: `39521231`

## Artifacts
- run_meta: `results/convnext_tiny_sr_v1_260130_104646/run_meta.json`
- config: `results/convnext_tiny_sr_v1_260130_104646/config.json`
- console_log: `results/convnext_tiny_sr_v1_260130_104646/console.log`
