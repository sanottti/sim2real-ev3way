# Phase 4 Stage 1 — 忠実化Sim + オモリ玉、難易度上限0.5(2026-10-01)

Phase 0〜3(単位誤読の訂正、Sim/app.c同期復元、8項目の実機忠実化、評価系の
信頼性回復)を経て、忠実化したSim上でv7から再学習した最初の成功例。

## 学習条件

- `python3 ev3way_train_run.py --w1 ... --w2 ... --popsize 150 --generations 150 --sigma0-resume 0.2 --stagnation-limit 60 --restart-sigma 0.3 --workers 6 --curriculum-gens 80 --max-difficulty 0.5 --checkpoint-every 5 --fidelity-all --weight-ball --train-seconds 5.0`
- v7(`known_good/v7_com_20260923/`)からwarm start
- 忠実化フラグ8項目すべて有効(モータ速度飽和・ジャイロ推定器・出力経路・
  制御周期5ms・転倒閾値45°・量子化+デッドバンド・起動デッドタイム・
  電池サグ)
- オモリ玉(`--weight-ball`)有効
- カリキュラムの難易度上限を1.0ではなく0.5に制限(`--max-difficulty`、
  今回新規実装)。理由: v7をdifficulty別に実測したところ0.5を境に崖があり
  (0.5→70%成功、1.0→35%成功)、フル難易度では個体差の勾配がほぼ消えて
  CMA-ESが学習できなかったため
- 150世代フル完走、自動リスタート0回(旧来のbest_flat異難易度比較バグを
  修正済みのため、カリキュラム中の誤ったリスタートが起きなくなった)

## 結果(50シード、fidelity-all、フル難易度での最終評価)

| | v7(未学習) | Stage1最終(gen150) |
|---|---|---|
| 平均生存時間 | 8.86s | **13.58s** |
| 中央値 | 0.66s | **1.00s** |
| 30秒完走率 | 28%(14/50) | **44%(22/50)** |
| 1秒未満で失敗 | 64%(32/50) | **50%(25/50)** |

学習対象だった難易度0.5では更に差が大きい(evaluate()スコア: v7=-1.060,
Stage1=-0.400)。学習していないフル難易度(1.0)でも上記の通り明確に改善。

## このセッションで踏んだ5つのトラブルと修正(詳細はexperiments.md参照)

1. `caffeinate`忘れでOSスリープにより1世代最大96分に悪化 → 独立起動で解消
2. 速度ペナルティがmax_speed=115rad/s時代の校正のまま → 正規化速度の二乗に修正
3. 学習中エピソード長15秒が忠実化後は過大(1世代8時間超) → `--train-seconds`追加
4. difficulty=1.0では個体差の勾配がほぼ消える → `--max-difficulty`追加
5. **本丸**: best_flatの「これまでの最良」判定が異なるdifficulty間で
   生スコア比較していた致命的バグ → カリキュラム中は常に直近世代を
   採用する方式に修正(この修正がなければ1〜4を直してもgen1の「たまたま
   易しい」個体が永久にbest_flatとして居座り続けていた)

## 次の一手(Stage 2)

Stage 1の結果からwarm startし、`--max-difficulty`を引き上げて(1.0に向けて)
再学習する。

## 復元手順

```bash
cd "Sim2RealEV3"
cp known_good/p4_stage1_ball_20261001/ev3way_w1.npy .
cp known_good/p4_stage1_ball_20261001/ev3way_w2.npy .
```
