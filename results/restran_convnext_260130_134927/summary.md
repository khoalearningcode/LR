# Run summary

- run_id: `restran_convnext_260130_134927`
- started: `2026-01-30T13:49:27+07:00`
- ended: `2026-01-30T14:01:13+07:00`
- duration_sec: `705.63`
- command: `train.py --model restran --backbone convnext --experiment-name restran_convnext --num-workers 8`

## Config
- EXPERIMENT_NAME: `restran_convnext`
- MODEL_TYPE: `restran`
- USE_STN: `True`
- DATA_ROOT: `data/train`
- EPOCHS: `30`
- BATCH_SIZE: `64`
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
- total_params: `18258348`
- trainable_params: `18258348`

## Artifacts
- run_meta: `results/restran_convnext_260130_134927/run_meta.json`
- config: `results/restran_convnext_260130_134927/config.json`
- console_log: `results/restran_convnext_260130_134927/console.log`
