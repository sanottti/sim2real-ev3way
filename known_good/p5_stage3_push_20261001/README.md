# Phase 5 Stage 3 — 押し外乱ロバスト性の追加学習(2026-10-01)

Phase 4 Stage2(`known_good/p4_stage2_ball_20261001/`)の結果からwarm start。
Simに新規実装した押し外乱(`p.applyExternalForce()`、エピソード中ランダムな
タイミング・強さで水平方向の力を0.1秒間加える)への耐性を学習した。

## 背景

ユーザーから「人が手で押しても倒れないロバスト性は確保できるか」と問われ、
**完全な押し耐性(どんな力でも倒れない)は物理的に不可能**(軽量・低トルク
モータの制約上、十分強い力なら必ず倒れる)が、**現実的な強さの小突きに
耐えられるようにする**ことは妥当な目標と回答。2026-09-22に一度試して
難しすぎたため撤去した押し外乱を、Stage1/2で機能した「難易度キャップ
戦略」を踏襲して再実装・再挑戦した。

## 実装

- `EV3WayEnv`に`PUSH_FORCE_MAX`(Newton、CLIは`--max-push-force`)を新設。
  reset()でエピソードごとに押しの強さ(scaled_uniformでdifficultyに応じて
  縮小、force_worst時は常に最大強度)とタイミング(0.5〜2.0秒の間で
  ランダム)を決定し、step()内で`p.applyExternalForce()`により水平方向の
  力を0.1秒間加える
- デフォルト0.0(無効)で既存の全実験と互換。`_init_worker`のinitargsにも
  伝搬済み

## 学習条件

- `python3 ev3way_train_run.py --w1 ... --w2 ... --popsize 150 --generations 150 --sigma0-resume 0.2 --stagnation-limit 60 --restart-sigma 0.3 --workers 6 --curriculum-gens 80 --max-difficulty 1.0 --max-push-force 3.0 --checkpoint-every 5 --fidelity-all --weight-ball --train-seconds 5.0`
- Stage2の最終重みからwarm start。150世代フル完走、自動リスタート0回、
  所要140.4分
- 押しの強さ上限3.0N(未実測のengineering estimate)。通常のドメイン
  ランダム化(difficulty)も0.2→1.0へ再度ランプさせているが、Stage2で
  既に習熟済みのため実質的な負荷はpushの新規学習が中心

## 結果(50シード、fidelity-all、フル難易度)

| | Stage2 | **Stage3(本版)** |
|---|---|---|
| push込み・平均生存 | 9.50s | **11.84s** |
| push込み・中央値 | 0.76s | **1.41s** |
| push込み・30秒完走 | 30%(15/50) | **38%(19/50)** |
| push無し・平均生存 | 15.20s | **18.70s** |
| push無し・中央値 | 15.31s | **30.00s** |
| push無し・30秒完走 | 50%(25/50) | **62%(31/50)** |

**push耐性が向上しただけでなく、push無しの通常条件でも退行せずさらに
改善**(オモリ玉+忠実化Simでの頑健性向上と相乗効果があったとみられる)。

evaluate()の内部スコア(push=3.0N・WORST_CASE_SEED込み・フル難易度)では
Stage2(-2.43)の方がStage3(-2.71)よりやや良い値だが、これはWORST_CASE_SEED
が常に最大強度のpushを含む単一の極端ケースであり、50シード統計(実用上
重視すべき指標)では明確にStage3が優れている。

## 復元手順

```bash
cd "Sim2RealEV3"
cp known_good/p5_stage3_push_20261001/ev3way_w1.npy .
cp known_good/p5_stage3_push_20261001/ev3way_w2.npy .
```

## 次の一手

- さらに押しの上限を引き上げるStage 3b、または複合最悪条件(WORST_CASE_SEED)
  への対応強化を検討
- 実機で実際にどの程度の押しに耐えられるか確認(Simのpush_force単位[N]は
  未較正のため、実機での定性的な確認が重要)
