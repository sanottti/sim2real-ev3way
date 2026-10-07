# ロバスト化Sim Stage1(2026-10-07)

`--calib-robust`(実測質量838g、慣性0.7〜3.5倍、重心幅2倍、KT0.08〜0.17、回転数16〜26rad/s、
遅れ15〜25ms、一次遅れ5〜12ms、不感帯0.05、初期傾き±8°)+`--fidelity-all`のSimで、v7とv16から再学習。

- `python3 ev3way_train_run.py --popsize 150 --generations 150 --sigma0-resume 0.2 --stagnation-limit 60 --restart-sigma 0.3 --workers 4 --curriculum-gens 80 --max-difficulty 0.5 --checkpoint-every 5 --fidelity-all --calib-robust --train-seconds 5.0`
- `*_v7.npy`: v7からwarm start(→app.c v17)、`*_v16.npy`: v16からwarm start

## ロバストSim 100シード評価(seed 300000〜)

| | 平均生存 | 30秒完走 | 飽和率 | \|PWM\|平均 |
|---|---|---|---|---|
| v7(学習前) | 4.86s | 15/100 | 1.0% | 38.4 |
| v16(学習前) | 3.40s | 10/100 | 9.4% | 16.3 |
| rob_v7 | 11.88s | 39/100 | 36.7% | 58.0 |
| rob_v16 | 10.83s | 35/100 | 30.4% | 47.3 |

注意: 生存時間は伸びたが飽和率が実機で悪かったv15(41%)並みに悪化。実機でSim予測が当たるかの検証用に
v17としてapp.cへ反映(rob_v7)。飽和ペナルティ付きStage2は別途学習中(`--sat-penalty 1.0`)。
