# 環境構築

## 必要なもの

- Python 3.12(開発時 3.12.2)
- macOS Apple Silicon で開発(Linux でも動くはずだが未検証)
- 実機側: LEGO MINDSTORMS EV3(ETロボコン EV3way-ET 構成)+ EV3RT

## 手順

```bash
python3 -m venv venv
source venv/bin/activate
pip install cma numpy matplotlib imageio imageio-ffmpeg
pip install pybullet     # ← macOS では失敗する可能性が高い。下記参照
```

学習スクリプトは起動時に `_ensure_installed()` で不足パッケージを自動
インストールしますが、pybullet だけは下記の理由で手動対応が必要な場合があります。

---

## ★ pybullet が macOS でビルドできない問題

PyPI の `pybullet` は macOS 向けのビルド済み wheel を配布しておらず、
`pip install pybullet` は毎回ソースからビルドされます。
新しい macOS + Xcode Command Line Tools の組み合わせでは、pybullet 同梱の
古い zlib コード(`examples/ThirdPartyLibs/zlib/zutil.h`)が原因で
コンパイルが失敗します。

```c
/* zutil.h の問題箇所 — 30年前のクラシックMac OS向けの分岐 */
#if defined(MACOS) || defined(TARGET_OS_MAC)
#ifndef fdopen
#define fdopen(fd, mode) NULL /* No fdopen() */
#endif
#endif
```

現代の macOS/Darwin でも `TARGET_OS_MAC` が真になるため、この分岐が誤って
有効化され、`<stdio.h>` が宣言する本物の `fdopen()` のシグネチャを破壊します。

```
_stdio.h:322: error: expected identifier or '('
```

### 対処法

sdist を取得し、`zutil.h` の当該分岐を `!defined(__APPLE__)` の場合のみ
有効になるようパッチしてから wheel をビルドします。

```bash
# 1. ソースを取得
pip download pybullet --no-binary :all: --no-deps -d /tmp/pb
cd /tmp/pb && tar xzf pybullet-3.2.7.tar.gz && cd pybullet-3.2.7

# 2. zutil.h をパッチ(該当の #if を __APPLE__ 以外に限定する)
#    examples/ThirdPartyLibs/zlib/zutil.h の
#      #if defined(MACOS) || defined(TARGET_OS_MAC)
#    を
#      #if (defined(MACOS) || defined(TARGET_OS_MAC)) && !defined(__APPLE__)
#    に書き換える

# 3. wheel をビルド
pip wheel . -w /tmp/pb/dist

# 4. インストール
pip install --no-deps /tmp/pb/dist/pybullet-*.whl
```

ビルド済み wheel はリポジトリには含めていません(68MB、環境依存バイナリの
ため)。オリジナルの開発環境では `vendor/` に保存してあります。

Python のバージョンを変えたり OS をアップデートした後に同じエラーが出たら、
上記手順を繰り返してください。

---

## 動作確認

```bash
source venv/bin/activate

# 1. Simとapp.cの数値一致(最重要・最初に実行する)
python3 test_sim_matches_appc.py
# → "PASS: Simはapp.cと数学的に一致しています"

# 2. 短い評価を回してみる
python3 benchmark_100seed.py --seeds 10 --fidelity-all \
  --w1 known_good/p5_stage3_push_20261001/ev3way_w1.npy \
  --w2 known_good/p5_stage3_push_20261001/ev3way_w2.npy
```

---

## 長時間学習を落とさないための設定(重要)

2つの独立した問題があり、**両方の対策が必要**です。

```bash
# 必ずこの形で起動する
nohup caffeinate -ims python3 ev3way_train_run.py ... > train.log 2>&1 &
disown
```

- `nohup ... & disown`: CLI のバックグラウンドタスク管理による約30分での
  強制終了を回避(`ps -o pid,ppid -p <PID>` で PPID=1 を確認)
- `caffeinate -ims`: macOS のスリープを防止
  (これを忘れて夜間に何度もスリープし、1世代が最大96分に悪化した実績あり)

起動前チェック:

```bash
pmset -g batt                     # AC電源に接続されているか
pmset -g | grep lowpowermode      # 0(オフ)であること
```

**Low Power Mode が ON だと1世代の所要時間が最大48倍に悪化します。**

学習中の進捗確認:

```bash
tail -f train.log
pmset -g assertions | grep caffeinate   # スリープ防止が効いているか
```
