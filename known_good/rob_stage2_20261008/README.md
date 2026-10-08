# ロバスト化Sim Stage2(2026-10-08)

rob_v7(Stage1)からwarm start。`--max-difficulty 1.0 --curriculum-gens 100 --popsize 150 --generations 150 --fidelity-all --calib-robust --train-seconds 5.0`

- `*_rob2_v7.npy`: `--sat-penalty 1.0`あり / `*_rob3_v7.npy`: ペナルティなし

## ロバストSim 100シード評価(seed 300000〜)

| | デッドタイムあり 平均生存/30秒完走/飽和率 | デッドタイムなし 平均生存/30秒完走/飽和率 |
|---|---|---|
| rob_v7(v17) | 11.88s / 39 / 36.7% | 21.73s / 72 / 19.4% |
| rob2_v7(ペナルティ) | 10.69s / 35 / 4.4% | 20.31s / 67 / 7.8% |
| rob3_v7(なし) | 11.17s / 35 / 22.8% | 20.44s / 64 / 22.5% |

結論: Stage2はStage1(v17)を超えない。ペナルティは飽和率だけを下げ、生存は改善しない。
v18候補にはしない。`python3 bench_robust_compare.py [--no-deadtime]`で再現。
