# Sim2Real EV3way-ET — PyBullet + CMA-ES で学習した倒立振子制御器

ETロボコン用 EV3way-ET(2輪倒立振子)を、PyBullet 上の強化学習(CMA-ES)で
学習し、実機 EV3(EV3RT / C言語)へ重みを移植するプロジェクトです。

**このリポジトリは引き継ぎ用です。** 別の人・別のLLM CLIが、途中から同じ
精度向上作業を継続できるよう、コード・学習済み重み・実機ログ・失敗を含む
全実験履歴を収録しています。

---

## 30秒でわかる現状(2026-10-05時点)

| 項目 | 状態 |
|---|---|
| 実機投入版 | **v16**(`nnapp/app.c`、`known_good/p5_stage3_push_20261001/`) |
| 実機での最高記録 | **v7 の平均生存 2.27秒**(9本、2026-09-29) |
| 直近の実機結果 | **v15 は平均1.49秒でv7より悪化**(5本、2026-10-01) |
| v16 の実機結果 | **未テスト ← 次にやること** |
| Sim上の最高 | v16(押し外乱学習済み) |

### いま最も重要な発見

**Sim上の生存時間だけで実機性能を判断してはいけません。**
v15 は Sim で v7 を大きく上回った(18.16s vs 11.16s)のに、実機では逆に
悪化しました(1.49s vs 2.27s)。

一方で **PWM飽和率は実機性能をよく予測します**:

| | Sim飽和率 | 実機飽和率 | 実機生存時間 |
|---|---|---|---|
| v7 | 7.5% | **2.0%** | **2.27s**(最良) |
| v15 | 38.2% | **41.1%** | 1.49s(最悪) |
| v16 | **2.2%** | 未測定 | **未測定** |

Sim は v15 の「モータを全力で叩く bang-bang 制御」を正確に予測できていた
(38.2% vs 41.1%)ので、物理忠実化そのものは機能しています。
v16 は Sim 上で飽和率 2.2% と、実機で成功した v7 の特徴に近い穏やかな制御を
獲得しており、**実機で良い結果が出る可能性があります**。

→ 詳細: [`real_logs/v15_20261001/analysis.md`](real_logs/v15_20261001/analysis.md)

---

## 次にやるべきこと(優先順)

1. **v16 を実機テストする**(最優先)
   - `nnapp/app.c` は既に v16。実機に書き込んで5〜10本ログを取る
   - LCD に `nnapp NN:v16` と表示されることを必ず確認する(後述の事故防止)
   - 期待: Sim予測の低飽和率(2.2%)が正しければ v7 (2.27s) を超える
2. **v16 も振るわない場合: モータトルクモデルの実測較正**
   - 現状 Sim の `eff_torque*(pwm - ω/max_speed)` は低速時に PWM100% で
     最大トルクが出る前提。実機は電流制限・H-bridge抵抗でそこまで出ないはず
   - 車輪を浮かせて PWM を段階的に与え、エンコーダから トルク-速度特性を実測する
3. **報酬に PWM飽和ペナルティを追加**して穏やかな制御を明示的に促す

---

## ドキュメント地図

| ファイル | 内容 |
|---|---|
| [`docs/SETUP.md`](docs/SETUP.md) | 環境構築(**pybullet が macOS で入らない問題の対処を含む**) |
| [`docs/DEPLOY_TO_EV3.md`](docs/DEPLOY_TO_EV3.md) | 実機への書き込み手順とログ回収、**バージョン取り違え事故の防止策** |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Sim と実機の対応関係、観測・報酬・忠実化フラグの設計 |
| [`docs/LESSONS.md`](docs/LESSONS.md) | **これまで踏んだ全バグと教訓**(同じ罠を踏まないために最重要) |
| `experiments.md` | 全実験の生ログ(時系列、113KB、失敗も全部残してある) |
| `SETUP_NOTES.md` | 環境構築時の詳細な作業記録 |
| `HANDOFF_SIM2REAL.md` | プロジェクト発足時の元の引き継ぎ資料 |
| `known_good/*/README.md` | 各バージョンの学習条件・結果・復元手順 |
| `real_logs/*/analysis.md` | 実機ログの解析結果 |

**初めて読む人は `docs/LESSONS.md` を先に読んでください。** このプロジェクトは
単位誤読・Sim/実機の設定ズレ・可視化バグなど、計測を誤らせるバグを何度も
踏んでおり、それを知らないと同じ結論の誤りを再生産します。

---

## リポジトリ構成

```
├── ev3way_train_run.py       # ★メインの学習スクリプト(CMA-ES)
├── ev3way_train_ppo.py       # PPO版(試したが改善せず)
├── ev3way_train_sac.py       # SAC版(試したが改善せず)
├── benchmark_100seed.py      # ★固定100シード評価(バージョン比較の正本)
├── phase2_cumulative_bench.py# 忠実化8項目の累積効果測定
├── test_sim_matches_appc.py  # ★Simとapp.cの数値一致を保証する回帰テスト
├── nnapp/app.c               # ★実機用Cコード(現在v16)
├── known_good/               # 各バージョンの重み・動画・README(正本)
├── real_logs/                # 実機ログ原本と解析
├── docs/                     # 引き継ぎドキュメント
└── experiments.md            # 全実験履歴
```

---

## クイックスタート

```bash
# 1. 環境構築(詳細は docs/SETUP.md、pybulletに注意)
python3 -m venv venv && source venv/bin/activate
pip install cma numpy matplotlib imageio imageio-ffmpeg
# pybullet は環境依存。docs/SETUP.md を必ず読むこと

# 2. Simとapp.cが一致していることを確認(最初に必ず実行)
python3 test_sim_matches_appc.py
# → "PASS: Simはapp.cと数学的に一致しています" が出ればOK

# 3. 現在の実機投入版(v16)の性能を測る
python3 benchmark_100seed.py --seeds 50 --fidelity-all \
  --w1 known_good/p5_stage3_push_20261001/ev3way_w1.npy \
  --w2 known_good/p5_stage3_push_20261001/ev3way_w2.npy

# 4. 続きから学習する(例: v16からさらに押し耐性を上げる)
cp known_good/p5_stage3_push_20261001/ev3way_w1.npy ev3way_w1_next.npy
cp known_good/p5_stage3_push_20261001/ev3way_w2.npy ev3way_w2_next.npy
nohup caffeinate -ims python3 ev3way_train_run.py \
  --w1 ev3way_w1_next.npy --w2 ev3way_w2_next.npy \
  --popsize 150 --generations 150 --workers 6 \
  --curriculum-gens 80 --max-difficulty 1.0 --max-push-force 4.0 \
  --fidelity-all --weight-ball --train-seconds 5.0 \
  > train_next.log 2>&1 &
disown
```

**長時間学習は `nohup caffeinate -ims ... & disown` が必須です**
(片方でも欠けると途中で止まります。理由は `docs/LESSONS.md`)。

---

## バージョン履歴(要約)

| 版 | 内容 | Sim | 実機 |
|---|---|---|---|
| v6 | プッシュ外乱撤廃・隠れ層16 | 良 | 約2.9秒(4本) |
| **v7** | 重心位置ランダム化 | 良 | **2.27秒(9本)← 実機最高記録** |
| v8〜v14 | 報酬設計変更・隠れ層24・PPO・SAC等 | — | **比較結果は全て無効**(後述) |
| v15 | 忠実化Sim + オモリ玉で再学習 | 18.16s | 1.49秒(5本、v7より悪化) |
| **v16** | v15 + 押し外乱ロバスト性 | 最良 | **未テスト** |

**v8〜v14 が無効な理由**: 2026-09-29 に (1) 実機ログの単位誤読
(`gyro_ang_x10` は度×10ではなく**ラジアン×10**)と (2) Sim と app.c の設定
ズレ(`N_HID`・`MPOS_SCALE`)が発覚し、この期間の比較はすべて誤った前提の上に
あったと判明しました。詳細は `docs/LESSONS.md`。

---

## ライセンス / 帰属

- 学習・解析コードはこのプロジェクトのオリジナルです
- `nnapp/app.c` は EV3RT(TOPPERS/HRP3)のサンプルアプリを土台にしています
  (EV3RT 本体のライセンスに従ってください)
- 実機は LEGO MINDSTORMS EV3(ETロボコン EV3way-ET 構成)
