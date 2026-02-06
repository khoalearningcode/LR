# Training Logs

<table>
<thead>
  <tr>
    <th style="border: 1px solid white;">Tag</th>
    <th style="border: 1px solid white;">Command</th>
    <th style="border: 1px solid white;">Val Acc</th>
    <th style="border: 1px solid white;">Public test (with val)</th>
    <th style="border: 1px solid white;">Public test (without val)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="border: 1px solid white;">restran + stn baselin</td>
    <td style="border: 1px solid white;">—</td>
    <td style="border: 1px solid white;">77.9%</td>
    <td style="border: 1px solid white;">—</td>
    <td style="border: 1px solid white;">75.6%</td>
  </tr>
  <tr>
    <td style="border: 1px solid white;">lrSim035_fd02_w160</td>
    <td style="border: 1px solid white;">

```bash
python train.py -m restran --backbone convnext --aug-level full --img-width 160 --train-lr-sim-p 0.35 --frame-dropout 0.20 --run-tag lrSim035_fd02_w160 --batch-size 64
```

</td>
    <td style="border: 1px solid white;">78.88%</td>
    <td style="border: 1px solid white;">75.5%</td>
    <td style="border: 1px solid white;">—</td>
  </tr>
  <tr>
    <td style="border: 1px solid white;">E6_w192_lrSim035_fd020</td>
    <td style="border: 1px solid white;">

```bash
python train.py -m restran --backbone convnext --aug-level full --epochs 30 --img-width 192 --train-lr-sim-p 0.35 --frame-dropout 0.20 --run-tag E6_w192_lrSim035_fd020 --batch-size 32
```

</td>
    <td style="border: 1px solid white;"><strong>79.48%</strong></td>
    <td style="border: 1px solid white;"><strong>77%</strong></td>
    <td style="border: 1px solid white;"><strong>77%</strong></td>
  </tr>
  <tr>
    <td style="border: 1px solid white;">restran_convnext</td>
    <td style="border: 1px solid white;">

```bash
python train.py -m restran --backbone convnext --aug-level full --epochs 30 --run-tag restran_convnext --batch-size 256
```

</td>
    <td style="border: 1px solid white;">76.38%</td>
    <td style="border: 1px solid white;">74.9%</td>
    <td style="border: 1px solid white;">—</td>
  </tr>
  <tr>
    <td style="border: 1px solid white;">TIMM_CXv2T_o0_w192</td>
    <td style="border: 1px solid white;">

```bash
python train.py -m restran --backbone timm --timm-model convnextv2_tiny --timm-out-index 0 \
  --backbone-pretrained \
  --aug-level full --epochs 30 \
  --img-width 192 --train-lr-sim-p 0.35 --frame-dropout 0.20 \
  --input-norm imagenet \
  --lr 2e-4 \
  --run-tag TIMM_CXv2T_o0_w192 \
  --batch-size 32 --save-every-steps 200 --overwrite
```

</td>
    <td style="border: 1px solid white;">77.68%</td>
    <td style="border: 1px solid white;">—</td>
    <td style="border: 1px solid white;">—</td>
  </tr>
  <tr>
    <td style="border: 1px solid white;">TIMM_CXv2T_fcmae22kIn1k_o0_w192</td>
    <td style="border: 1px solid white;">

```bash
python train.py -m restran --backbone timm \
  --timm-model convnextv2_tiny.fcmae_ft_in22k_in1k --timm-out-index 0 \
  --backbone-pretrained --input-norm imagenet \
  --aug-level full --epochs 30 \
  --img-width 192 --train-lr-sim-p 0.35 --frame-dropout 0.20 \
  --lr 1.5e-4 \
  --run-tag TIMM_CXv2T_fcmae22kIn1k_o0_w192 \
  --batch-size 32 --save-every-steps 200 --overwrite
```

</td>
    <td style="border: 1px solid white;">76.68%</td>
    <td style="border: 1px solid white;">—</td>
    <td style="border: 1px solid white;">—</td>
  </tr>
</tbody>
</table>