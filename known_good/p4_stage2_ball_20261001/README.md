# Phase 4 Stage 2 — 忠実化Sim + オモリ玉、難易度上限1.0(2026-10-01)

Stage 1(`known_good/p4_stage1_ball_20261001/`)の結果からwarm startし、
カリキュラムの難易度上限を0.5→1.0(フル難易度)に引き上げて再学習。

## 学習条件

- `python3 ev3way_train_run.py --w1 ... --w2 ... --popsize 150 --generations 150 --sigma0-resume 0.2 --stagnation-limit 60 --restart-sigma 0.3 --workers 6 --curriculum-gens 100 --max-difficulty 1.0 --checkpoint-every 5 --fidelity-all --weight-ball --train-seconds 5.0`
- Stage 1の最終重みからwarm start
- 忠実化フラグ8項目・オモリ玉は引き続き有効
- 150世代フル完走、自動リスタート0回、所要134.2分

## 結果(50シード、fidelity-all、フル難易度)

| | v7(未学習) | Stage1 | **Stage2(本版)** |
|---|---|---|---|
| 平均生存時間 | 11.16s | 11.74s | **18.16s** |
| 中央値 | 0.66s | 0.67s | **30.00s** |
| 30秒完走率 | 36%(18/50) | 38%(19/50) | **60%(30/50)** |
| 1秒未満で失敗 | 60%(30/50) | 56%(28/50) | **40%(20/50)** |

**典型条件への頑健性はv7・Stage1を大きく上回る**。

## 注意点: 複合最悪条件では逆にv7よりやや劣る

`evaluate()`の内部スコア(WORST_CASE_SEEDを3倍重み付け)で比較すると:

| | v7 | Stage1 | Stage2 |
|---|---|---|---|
| スコア(difficulty=1.0) | **-2.541** | -2.733 | -2.612 |

v7 > Stage2 > Stage1 の順で、Stage2は「典型的な条件」には強いが
「最も過酷な複合最悪条件(重量増+重心高+摩擦低下+電池弱+トルク損失最大+
COM最大シフト)」への対応はv7よりわずかに後退している。実機投入前に
最悪条件寄りの追加学習(E1: マルチシード探索、または複合最悪条件の
重み増加)を検討する余地がある。

## 動画

`training_result_p4_ball.mp4`(最終世代150体、6.0秒、ランダム化なしの
公平条件で同時撮影)。Artifactとしても公開済み:
https://claude.ai/code/artifact/65a6740b-006a-44df-bd5b-f122a97d2528

★2026-10-01: 初回公開時、`record_population_video()`の3つのバグ
(忠実化フラグ未反映・オモリ玉が範囲外の固定位置・物理タイムステップが
5ms化に未追従で起動デッドタイムが実質2倍実行)により、実際には安定して
いるはずの重みが動画上では1秒以内に全個体転倒して見えるという不具合が
あった。ユーザー指摘により発見・修正済み(詳細はexperiments.md参照)。
現在のファイルは修正後の版。

## 復元手順

```bash
cd "Sim2RealEV3"
cp known_good/p4_stage2_ball_20261001/ev3way_w1.npy .
cp known_good/p4_stage2_ball_20261001/ev3way_w2.npy .
```

## 次の一手

1. 複合最悪条件への追加チューニング(E1マルチシード探索 or A案の重み増加)
2. それを経ずにこのままPhase 5(実機検証)へ進める選択肢もある
   (典型条件での頑健性が大幅改善しているため、実用上は十分に価値がある)
