# v6 (push外乱なし) — 現行ベストのスナップショット

2026-09-22 取得。**この状態を「悪くなったら戻る場所」として保全する。**
以降の実験(プッシュ外乱の再導入など)でメインの`ev3way_w1.npy`/`w2.npy`/
`training_result.mp4`が上書きされても、このフォルダの中身は変更しない。

## 学習条件
- popsize=200, generations=30(gen15で早期終了、所要50分)
- 隠れ層16ユニット、TRAIN_MAX_STEPS=1500(15秒)、**前後プッシュ外乱なし**
- warm startから新規学習

## 結果
- best_reward = 1479.3(理論上限相当)
- 通常条件での生存: 4.6〜8.8秒
- 最悪条件(重量増+重心高+摩擦低下+電池弱+トルク損失最大、プッシュなし): **30.0秒完走**(最大傾き3.0°)
- 既知の弱点: プッシュ外乱を学習していないため、実機で手で押すような外乱には弱い可能性が高い

## 実機投入
`app_c_snapshot.c`が、2026-09-22時点で`nnapp/app.c`として
実機投入された内容と同一(W1[7][16]/W2[16][2]、nn_forward()のループ上限・
h[]サイズも16に更新済み)。

## 元に戻す手順(この状態へロールバックする場合)

```bash
cd "Sim2RealEV3"
cp known_good/v6_pushfree_20260922/ev3way_w1.npy .
cp known_good/v6_pushfree_20260922/ev3way_w2.npy .
cp known_good/v6_pushfree_20260922/training_result.mp4 .
cp known_good/v6_pushfree_20260922/app_c_snapshot.c nnapp/app.c
```
