# 実機(EV3)への書き込みとログ回収

## 全体の流れ

```
学習(PyBullet/CMA-ES)
  → ev3way_w1.npy / ev3way_w2.npy
  → C配列に変換して nnapp/app.c の W1[7][16] / W2[16][2] を差し替え
  → NN_VERSION を更新
  → test_sim_matches_appc.py で一致確認
  → EV3RT でビルドして実機へ転送
  → 実機で走らせる(ログがSDカードに溜まる)
  → /Volumes/EV3RT/nn_<version>_NNN.csv を回収して解析
```

---

## 1. 学習済み重みを app.c に反映する

```bash
source venv/bin/activate
python3 -c "
import numpy as np, ev3way_train_run as sim
w1 = np.load('known_good/p5_stage3_push_20261001/ev3way_w1.npy')
w2 = np.load('known_good/p5_stage3_push_20261001/ev3way_w2.npy')
print(sim.to_c('W1', w1)); print(); print(sim.to_c('W2', w2))
"
```

出力された2つの配列定義で `nnapp/app.c` の `W1`/`W2` を丸ごと置き換えます。

**あわせて必ず `NN_VERSION` を更新してください**:

```c
#define NN_VERSION "v17"   /* ← 重みを変えたら必ず上げる */
```

### 反映後の必須チェック

```bash
# test_sim_matches_appc.py の参照先を新しい known_good ディレクトリに変えてから
python3 test_sim_matches_appc.py
# → "PASS: Simはapp.cと数学的に一致しています" が出るまで実機に書かない
```

このテストは app.c の `W1`/`W2` リテラルと正規化定数(`MPOS_SCALE` 等)を
パースし、Sim の `nn_forward()` と出力が一致することを検証します。
**隠れ層サイズを変えた場合は `nn_forward()` のループ上限と `h[]` の
配列サイズも変更が必要です**(現在は16で固定)。

---

## 2. EV3 実機へのビルド・転送

実機側は EV3RT(TOPPERS/HRP3)環境です。`nnapp/app.c` を EV3RT のアプリとして
ビルドし、BeerHall 等のツールで EV3 本体へ転送します。

> **このリポジトリには EV3RT のビルド環境自体は含まれていません。**
> 以下は、元の開発者(macOS / Apple Silicon)が実機で実際に行った手順です。
> 環境構築そのもの(Xcode CLT、Rosetta、etrobo の初回インストール)は
> ETロボコン公式の etrobo 手順に従ってください。

### 2-1. 前提(一度だけ)

| 項目 | 内容 |
|---|---|
| ビルド環境 | etrobo(BeerHall = サンドボックス Homebrew 上の環境) |
| `$BEERHALL` | `~/Desktop/BeerHall/` |
| `$ETROBO_ROOT` | `<BeerHall>/etrobo`(例: `/Users/<user>/Desktop/BeerHall/etrobo`) |
| クロスコンパイラ | `gcc-arm-none-eabi-6-2017-q1`(自動導入されなかったため、Arm の配布サイトから手動で導入した) |
| SDカード | 32GB、**FAT32**、**ボリューム名 `EV3RT`**(Mac 上では `/dev/disk6` だった。番号は環境で変わる) |
| ローダ | `ev3 install` で uImage を SD に書込み済み |
| アプリ転送先 | `/Volumes/EV3RT/ev3rt/apps/` |

ボリューム名が `EV3RT` で始まっていないと、`make ... up` の自動転送が効きません。

### 2-2. etrobo 環境に入る

```bash
cd "$BEERHALL" && ./BeerHall
```

BeerHall の起動がスキップされる場合は、`~/.zprofile` に残っている
BEERHALL 関連の行(過去の残骸)を削除してから入り直します。

### 2-3. プロジェクトを作る(初回のみ)

```bash
cd "$ETROBO_ROOT/workspace"
mkdir -p nnapp && cp sample_c4/Makefile.inc nnapp/
# あるいは丸ごとコピー: cp -r sample_c4 nnapp
```

その後、`app.c` / `app.cfg` / `app.h` を `workspace/nnapp/` に置きます。
このリポジトリの `nnapp/` には `app.c` しかないため、`app.cfg` と `app.h` は
`sample_c4` のものをベースに用意してください。
`app.cfg` の `CRE_CYC` で構文エラーが出た場合は、
`workspace/periodic-task/app.cfg` の書式と見比べて直します。

**プロジェクト名に `app` を使わないでください。** workspace 内の既存ファイル
(ビルド済みバイナリ)と衝突し、
`cp: 'app/app.c' を stat できません: Not a directory` になります。
`rm -f app` で消すか、`nnapp` / `ev3app` / `calib` のような別名にします。
また、ダウンロードした `app.c` / `app.h` / `app.cfg` はプロジェクトごとに
同名になるため、「ダウンロード → 配置」を1プロジェクトずつ順に行ってください。

### 2-4. ビルドして SD に転送する

SD カードを Mac に挿した状態で実行します。

```bash
cd "$ETROBO_ROOT" && make app=nnapp up 2>&1 | tail -3
```

成功すると次の2行が出ます。

```
'nnapp' is copied into EV3.
fakemake on Uhrp3: build succeed: nnapp
```

これらが出ない場合は SD に書き込まれていません(ソースを更新しただけでは
実機は変わらない、§3 を参照)。

### 2-5. SD を取り出して実機にセットする

```bash
diskutil eject /Volumes/EV3RT
```

1. SD を EV3 に挿して電源を ON にする
2. ローダメニューで上下ボタンにより `SD card` → 対象プログラムを選ぶ
   (開発中は `calib`(診断)、`ev3app`(gyroboy 線形制御)、`nnapp`(NN 制御)が
   同じ階層に並んでいた)
3. LCD 最上段の `nnapp NN:vXX` が期待どおりか確認する(§3)

実機での操作手順は §4、ログの回収は §5 を参照してください。

### 2-6. 動作確認済み / 未検証

| 項目 | 状態 |
|---|---|
| macOS(Apple Silicon)+ BeerHall でのビルド・SD転送 | **実機で確認済み** |
| Windows 11 + WSL2 でのビルド・SD転送 | **未検証**(`HANDOFF_SIM2REAL.md` に手順案はあるが、実機では未確認) |

---

## 3. ★ バージョン取り違えを防ぐ(過去に事故あり)

**「app.c のソースを更新した」ことと「実機に書き込まれている」ことは別です。**

実際に、ソースが v8→v9 と進む間、実機には数日前の v7 が入ったままで、
それに気づかず「v9の実機テスト結果」として記録・解析してしまう事故が
起きました(後日、既知の重みで観測値から出力を再現し実測PWMとの一致度を
比較することで v7 と判明)。

再発防止として、現在は**同じバージョン文字列が4箇所に出る**ようにしています:

| 場所 | 形式 |
|---|---|
| ソース | `#define NN_VERSION "v16"` |
| **LCD画面(起動時)** | `nnapp NN:v16` |
| ログファイル名 | `nn_v16_000.csv` |
| CSVヘッダー | `# nn_weights_version=v16` |

**実機テストの前に、必ず LCD の表示でバージョンを目視確認してください。**
これが人間が気づける唯一の砦です。

---

## 4. 実機での走らせ方

1. ロボットを平らな床に置く(尻尾で自立させる)
2. 起動すると LCD 最上段に `nnapp NN:vXX`、中段に `PUSH TOUCH` が表示される
3. **LCD のバージョン表示を確認**
4. タッチセンサを押して離す(`PUSH TOUCH` → `RELEASE`)
5. ジャイロ校正が始まる(`calibrating...` / `hold still`)。**約2.5秒、機体に触らない**。
   LCD に `ofs`/`rng` と品質(`GOOD` / `so-so` / `BAD!`)が出る(表示のみで、必ず走行可能になる)
6. `ready! (NN)` / `touch: GO` が出たらタッチセンサを押す。0.5秒後に尻尾モータが解放され倒立制御が始まる
7. **必ず手を離して自力で立つか確認する**(手で支えたテストは判断材料にならない)
8. 転倒(傾き45°超)またはタッチセンサ再押下で停止し、ログが保存される
   (タッチ停止時は PWM を約100msかけて減速してからモータを止める。転倒検知時は即停止)

### 制御ループの要点(app.c)

| 項目 | 値 |
|---|---|
| 制御周期 | 約5ms(`WAIT_TIME_MS 5`) |
| 転倒判定 | `FALL_ANGLE_DEG 45.0f` |
| 出力ゲイン | `OUTPUT_GAIN 1.5f` |
| ジャイロのオフセット補正 | EMA(`EMAOFFSET 0.0005f`) |
| 左右出力 | **平均化して同じ値を両輪に出す**(旋回は制御していない) |

`OUTPUT_GAIN` を変える場合は、**Sim 側の `OUTPUT_GAIN` 定数
(`ev3way_train_run.py`)も合わせて変更し、その値で再学習してください**。
実機だけ変えると Sim とズレて性能が保証されません。

---

## 5. ログの回収と解析

ログは EV3 の SD カード(macOS では `/Volumes/EV3RT/`)に
`nn_<version>_000.csv`, `nn_<version>_001.csv`, ... と溜まります。

```bash
# 回収(リポジトリに永久保存する)
mkdir -p real_logs/v16_YYYYMMDD
cp /Volumes/EV3RT/nn_v16_*.csv real_logs/v16_YYYYMMDD/
```

### CSV の形式

```
# nnapp Sim2Real NN controller v3 (gain=1.50)
# nn_weights_version=v16
# gyro_ofs_mdps=-1043
# wait_ms=5,decim=2,samples=180
t_ms,gyro_raw,gyro_spd_x10,gyro_ang_x10,cnt_l,cnt_r,batt_mV,pwm_l,pwm_r
```

| 列 | 意味 | **単位(重要)** |
|---|---|---|
| `t_ms` | 経過時間 | ミリ秒 |
| `gyro_raw` | ジャイロ生値 | dps(整数) |
| `gyro_spd_x10` | 角速度 | **rad/s × 10** |
| `gyro_ang_x10` | 推定傾き角 | **★ラジアン × 10(度ではない)** |
| `cnt_l`/`cnt_r` | エンコーダ | 度 |
| `batt_mV` | 電池電圧 | mV |
| `pwm_l`/`pwm_r` | モータ出力 | -100〜100 |

**`gyro_ang_x10` をラジアンではなく度と誤読して3日間を無駄にした事故が
あります。** 記録値 `7` は `0.7°` ではなく `0.7rad ≈ 40°`(転倒閾値45°の直前)です。

### 解析テンプレート

```python
import csv, math, glob
import numpy as np

def load(path):
    with open(path) as f:
        lines = f.readlines()
    ds = next(j for j, l in enumerate(lines) if not l.startswith('#'))
    return [{k: float(v) for k, v in r.items()}
            for r in csv.DictReader(lines[ds:])]

for p in sorted(glob.glob('real_logs/v16_YYYYMMDD/nn_v16_*.csv')):
    rows = load(p)
    dur = rows[-1]['t_ms'] / 1000.0
    max_tilt = max(abs(r['gyro_ang_x10']) / 10.0 for r in rows)  # rad
    pw = np.array([r['pwm_l'] for r in rows])
    sat = np.mean(np.abs(pw) >= 95) * 100   # PWM飽和率[%]
    print(f'{p}: {dur:.2f}s  max_tilt={math.degrees(max_tilt):.1f}deg  '
          f'飽和率={sat:.1f}%  |PWM|平均={np.mean(np.abs(pw)):.1f}')
```

### 見るべき指標

| 指標 | 意味 | 基準 |
|---|---|---|
| 生存時間 | 転倒までの秒数 | v7 = 2.27秒(実機最高記録) |
| 最大/最終傾き | 45°近傍なら自動転倒 | — |
| **PWM飽和率** | **実機性能をよく予測する** | v7 = 2.0%(良)、v15 = 41.1%(悪) |
| \|PWM\|平均 | 制御の激しさ | v7 = 30.9、v15 = 60.1 |
| 車輪ドリフト | その場に留まれているか | — |

**PWM飽和率が高い = モータの性能限界を超えた指令を出し続けている**状態で、
実機では逆起電力・電流制限で実トルクが出ず制御が破綻します。
Sim 上でこの指標を事前に確認してから実機に投入するのが有効です。

```bash
# Sim上で飽和率を事前確認する例は real_logs/v15_20261001/analysis.md 参照
```
