# Run summary

- run_id: `restran_260129_231844`
- started: `2026-01-29T23:18:44+07:00`
- ended: `2026-01-30T00:14:56+07:00`
- duration_sec: `3371.97`
- command: `train.py`

## Config
- EXPERIMENT_NAME: `restran`
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
- total_params: `31077676`
- trainable_params: `31077676`

## Best checkpoint / metrics
- best_ckpt: `results/restran_260129_231844/restran_best.pth`

## Artifacts
- best_ckpt: `results/restran_260129_231844/restran_best.pth`
- run_meta: `results/restran_260129_231844/run_meta.json`
- config: `results/restran_260129_231844/config.json`
- console_log: `results/restran_260129_231844/console.log`
