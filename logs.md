# Training Logs

<table>
<thead>
  <tr>
    <th style="border: 1px solid white;">Version</th>
    <th style="border: 1px solid white;">Tag</th>
    <th style="border: 1px solid white;">Command</th>
    <th style="border: 1px solid white;">Val Acc</th>
    <th style="border: 1px solid white;">Public test (with val)</th>
    <th style="border: 1px solid white;">Public test (without val)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="4" style="border: 1px solid white;">v1</td>
    <td style="border: 1px solid white;">restran + stn baselin</td>
    <td style="border: 1px solid white;">—</td>
    <td style="border: 1px solid white;">77.9%</td>
    <td style="border: 1px solid white;">—</td>
    <td style="border: 1px solid white;">0.756</td>
  </tr>
  <tr>
    <td style="border: 1px solid white;">lrSim035_fd02_w160</td>
    <td style="border: 1px solid white;">

```bash
python train.py -m restran --backbone convnext --aug-level full --img-width 160 --train-lr-sim-p 0.35 --frame-dropout 0.20 --run-tag lrSim035_fd02_w160 --batch-size 64
```

</td>
    <td style="border: 1px solid white;">78.88%</td>
    <td style="border: 1px solid white;">0.755</td>
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
    <td style="border: 1px solid white;"><strong>0.77</strong></td>
    <td style="border: 1px solid white;"><strong>0.77</strong></td>
  </tr>
  <tr>
    <td style="border: 1px solid white;">restran_convnext</td>
    <td style="border: 1px solid white;">

```bash
python train.py -m restran --backbone convnext --aug-level full --epochs 30 --run-tag restran_convnext --batch-size 256
```

</td>
    <td style="border: 1px solid white;">76.38%</td>
    <td style="border: 1px solid white;">0.749</td>
    <td style="border: 1px solid white;">—</td>
  </tr>
</tbody>
</table>
