# デッドタイムなしSim Stage1再学習(2026-10-09)

`--fidelity-all --no-startup-deadtime --calib-robust`、v7/v16からwarm start、Stage1と同じ設定(150世代・popsize150・難易度0.5)。

ロバストSim(`bench_robust_compare.py --no-deadtime`)100シード(seed300000〜):

| | 平均生存 | 30秒完走 | 飽和率 |
|---|---|---|---|
| rob_v7(v17、デッドタイムありで学習) | 21.73s | 72 | 19.4% |
| nd_v7 | 17.38s | 57 | 30.5% |
| nd_v16 | 17.34s | 57 | 14.9% |

結論: デッドタイムなしで学習しても、デッドタイムありで学習したv17を超えない(1本ずつの試行でバラつきは未検証)。v18候補にしない。
