# Run summary

- run_id: `restran_260130_150726_lrSim035_fd02_w160`
- started: `2026-01-30T15:07:26+07:00`
- ended: `2026-01-30T16:42:55+07:00`
- duration_sec: `5729.2`
- command: `train.py -m restran --backbone convnext --aug-level full --img-width 160 --train-lr-sim-p 0.35 --frame-dropout 0.20 --run-tag lrSim035_fd02_w160`

## Config
- EXPERIMENT_NAME: `restran`
- MODEL_TYPE: `restran`
- USE_STN: `True`
- DATA_ROOT: `data/train`
- EPOCHS: `30`
- BATCH_SIZE: `64`
- LEARNING_RATE: `0.0005`
- IMG_HEIGHT: `32`
- IMG_WIDTH: `160`
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
- total_params: `37266124`
- trainable_params: `37266124`

## Best checkpoint / metrics
- best_ckpt: `results/restran_260130_150726_lrSim035_fd02_w160/restran_best.pth`

## Artifacts
- best_ckpt: `results/restran_260130_150726_lrSim035_fd02_w160/restran_best.pth`
- run_meta: `results/restran_260130_150726_lrSim035_fd02_w160/run_meta.json`
- config: `results/restran_260130_150726_lrSim035_fd02_w160/config.json`
- console_log: `results/restran_260130_150726_lrSim035_fd02_w160/console.log`
