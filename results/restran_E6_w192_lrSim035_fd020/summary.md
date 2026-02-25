# Run summary

- run_id: `restran_E6_w192_lrSim035_fd020`
- started: `None`
- ended: `2026-02-01T16:29:46+00:00`
- duration_sec: `5063.2`
- command: `None`

## Config
- EXPERIMENT_NAME: `restran`
- MODEL_TYPE: `restran`
- USE_STN: `True`
- DATA_ROOT: `/kaggle/input/icpr2026/data/train`
- EPOCHS: `30`
- BATCH_SIZE: `32`
- LEARNING_RATE: `0.0005`
- IMG_HEIGHT: `32`
- IMG_WIDTH: `192`
- AUGMENTATION_LEVEL: `full`
- VAL_SPLIT_FILE: `/kaggle/input/icpr2026/data/val_tracks.json`
- DEVICE: `cuda`

## Dataset
- train_size: `38002`
- val_size: `999`
- submission_mode: `False`
- data_root: `/kaggle/input/icpr2026/data/train`
- test_data_root: `/kaggle/input/icpr2026/data/public_test`

## Model
- type: `restran`
- use_stn: `True`
- total_params: `37266124`
- trainable_params: `37266124`

## Best checkpoint / metrics
- best_ckpt: `results/restran_E6_w192_lrSim035_fd020/restran_best.pth`

## Artifacts
- best_weights: `results/restran_E6_w192_lrSim035_fd020/restran_best.pth`
- last_weights: `results/restran_E6_w192_lrSim035_fd020/restran_last.pth`
- best_full_ckpt: `results/restran_E6_w192_lrSim035_fd020/checkpoints/best.pt`
- last_full_ckpt: `results/restran_E6_w192_lrSim035_fd020/checkpoints/last.pt`
- run_meta: `results/restran_E6_w192_lrSim035_fd020/run_meta.json`
- config: `results/restran_E6_w192_lrSim035_fd020/config.json`
- console_log: `results/restran_E6_w192_lrSim035_fd020/console.log`
